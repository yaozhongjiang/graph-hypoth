"""Tests for deterministic graph-delta validation and application.

The suite covers all five acceptance gates, pure application with no partial mutation,
family and base-hash rejection, idempotency, per-family transitions, and node merging.
It is LLM-free and deterministic.
"""

from __future__ import annotations

from src.graph_store import (
    CausalClaimGraphStore,
    EdgeStatus,
    EvidenceRole,
    ExperimentDesign,
    ExperimentPlan,
)
from src.delta import (
    AddEdgeOp,
    AddNodeOp,
    DeltaFamily,
    ExperimentPayload,
    ExtractPayload,
    GraphDeltaProposal,
    HypothesisPayload,
    MergeNodesOp,
    PriorityPayload,
    VerificationVerdict,
    VerifyPayload,
    build_edge,
    build_node,
)
from src.validator import GraphDeltaValidator


def _extract(operations, base_hash, **env) -> GraphDeltaProposal:
    return GraphDeltaProposal(
        family=DeltaFamily.EXTRACT, base_graph_hash=base_hash,
        payload=ExtractPayload(operations=operations), **env,
    )


def _seed_one_edge(status=EdgeStatus.UNVERIFIED):
    """A store holding one edge n1->n2 (+ its two nodes) at the given status."""
    store = CausalClaimGraphStore()
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="cancer", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    edge.status = status
    store.nodes[n1.node_id] = n1
    store.nodes[n2.node_id] = n2
    store.edges[edge.edge_id] = edge
    return store, edge


# --- Deterministic gate evaluation -----------------------------------------------------
def test_evaluate_is_deterministic_across_runs():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    delta = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], store.base_hash)
    first = v.evaluate(store, delta)
    second = v.evaluate(store, delta)
    assert (first.schema, first.refs, first.base_hash, first.allowed_transition, first.idempotent) == (
        second.schema, second.refs, second.base_hash, second.allowed_transition, second.idempotent
    )
    assert first.accept_i == second.accept_i == 1


# --- Pure application without partial mutation ----------------------------------------
def test_apply_is_pure_and_leaves_the_input_store_byte_identical():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    before = store.base_hash
    delta = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], store.base_hash)
    new_store = v.apply(store, delta)
    assert new_store is not store
    assert store.base_hash == before          # input untouched
    assert new_store.base_hash != before       # result advanced
    assert new_store.version == store.version + 1


def test_failing_gate_makes_accept_zero_with_no_apply_needed():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    delta = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], "WRONG_BASE_HASH")
    decision = v.evaluate(store, delta)
    assert decision.accepted is False
    assert decision.base_hash is False
    assert decision.failing_gate == "base_hash"


# --- Base-hash validation ---------------------------------------------------------------
def test_base_hash_mismatch_is_rejected():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    delta = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], "NOT_THE_BASE")
    assert v.evaluate(store, delta).accept_i == 0


def test_family_payload_mismatch_fails_the_schema_gate():
    store = CausalClaimGraphStore()
    # The family declares extraction but the payload is verification, so the schema gate must fail.
    delta = GraphDeltaProposal(
        family=DeltaFamily.EXTRACT, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id="e1", evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT),
    )
    decision = GraphDeltaValidator().evaluate(store, delta)
    assert decision.schema is False
    assert decision.failing_gate == "schema"


# --- Idempotency validation -------------------------------------------------------------
def test_idempotency_gate_rejects_an_already_committed_key():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    delta = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], store.base_hash)
    assert v.evaluate(store, delta, committed_keys=set()).idempotent is True
    assert v.evaluate(store, delta, committed_keys={delta.idempotency_key()}).idempotent is False


# --- Transition table by delta family --------------------------------------------------
def test_extract_add_edge_must_land_unverified():
    v = GraphDeltaValidator()
    # endpoints resolve, so only the transition gate is at stake.
    store = CausalClaimGraphStore()
    n1 = build_node(label="a", type="construct")
    n2 = build_node(label="b", type="outcome")
    store.nodes[n1.node_id] = n1
    store.nodes[n2.node_id] = n2
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    edge.status = EdgeStatus.SUPPORTED  # illegal landing status for extract
    delta = _extract([AddEdgeOp(edge=edge)], store.base_hash)
    assert v.evaluate(store, delta).allowed_transition is False


def test_verify_moves_unverified_to_settled_status():
    v = GraphDeltaValidator()
    store, edge = _seed_one_edge(EdgeStatus.UNVERIFIED)
    delta = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT, confidence=0.8),
    )
    assert v.evaluate(store, delta).accept_i == 1
    applied = v.apply(store, delta)
    assert applied.edges[edge.edge_id].status == EdgeStatus.SUPPORTED


def test_verify_cannot_change_a_terminal_status():
    v = GraphDeltaValidator()
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    delta = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.CONTRADICT),
    )
    assert v.evaluate(store, delta).allowed_transition is False


def test_verify_may_move_insufficient_to_settled():
    v = GraphDeltaValidator()
    store, edge = _seed_one_edge(EdgeStatus.INSUFFICIENT)
    delta = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT),
    )
    assert v.evaluate(store, delta).allowed_transition is True


def test_priority_delta_never_changes_status():
    v = GraphDeltaValidator()
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    node_id = next(iter(store.nodes))
    delta = GraphDeltaProposal(
        family=DeltaFamily.PRIORITY, base_graph_hash=store.base_hash,
        payload=PriorityPayload(node_or_edge_ids=[node_id], priority_values=[0.9]),
    )
    assert v.evaluate(store, delta).accept_i == 1
    applied = v.apply(store, delta)
    assert applied.edges[edge.edge_id].status == EdgeStatus.SUPPORTED  # untouched
    assert applied.nodes[node_id].user_priority == 0.9                  # priority applied


def test_priority_delta_targeting_an_edge_is_rejected_priority_is_node_only():
    # User priority lives on ConceptNode only; an edge target has no
    # place to store priority, so it must be rejected, never accepted-then-silently-dropped.
    v = GraphDeltaValidator()
    store, edge = _seed_one_edge(EdgeStatus.UNVERIFIED)
    delta = GraphDeltaProposal(
        family=DeltaFamily.PRIORITY, base_graph_hash=store.base_hash,
        payload=PriorityPayload(node_or_edge_ids=[edge.edge_id], priority_values=[0.9]),
    )
    decision = v.evaluate(store, delta)
    assert decision.refs is False
    assert decision.accept_i == 0


def test_priority_mismatched_id_value_lengths_fail_the_schema_gate():
    # A priority delta whose node_or_edge_ids and priority_values have unequal lengths must
    # fail schema validation; otherwise apply() would silently commit only the zipped prefix.
    v = GraphDeltaValidator()
    store, _edge = _seed_one_edge(EdgeStatus.UNVERIFIED)
    node_id = next(iter(store.nodes))
    delta = GraphDeltaProposal(
        family=DeltaFamily.PRIORITY, base_graph_hash=store.base_hash,
        payload=PriorityPayload(node_or_edge_ids=[node_id], priority_values=[0.9, 0.5]),
    )
    decision = v.evaluate(store, delta)
    assert decision.schema is False
    assert decision.failing_gate == "schema"


# --- Node merging ----------------------------------------------------------------------
def test_merge_nodes_redirects_edges_dedups_and_retires_merged():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    a = build_node(label="myocardial infarction", type="outcome")
    b = build_node(label="heart attack", type="outcome")
    c = build_node(label="aspirin", type="exposure/intervention")
    for n in (a, b, c):
        store.nodes[n.node_id] = n
    edge_ac = build_edge(source_node_ids=[c.node_id], target_node_ids=[a.node_id],
                         direction="causal", relation_type="reduces")
    edge_bc = build_edge(source_node_ids=[c.node_id], target_node_ids=[b.node_id],
                         direction="causal", relation_type="reduces")
    edge_ac.open_risks = ["risk-a"]
    edge_bc.open_risks = ["risk-b"]
    store.edges[edge_ac.edge_id] = edge_ac
    store.edges[edge_bc.edge_id] = edge_bc

    delta = _extract([MergeNodesOp(survivor_node_id=a.node_id, merged_node_id=b.node_id)],
                     store.base_hash)
    assert v.evaluate(store, delta).accept_i == 1
    applied = v.apply(store, delta)

    # merged node retired; survivor absorbs the merged label as an alias
    assert b.node_id not in applied.nodes
    assert a.node_id in applied.nodes
    assert "heart attack" in applied.nodes[a.node_id].aliases
    # c->a and (redirected) c->b collapse to a single c->a edge, unioning open_risks
    assert len(applied.edges) == 1
    survived = next(iter(applied.edges.values()))
    assert survived.target_node_ids == [a.node_id]
    assert set(survived.open_risks) == {"risk-a", "risk-b"}


def test_merge_collision_prefers_contradicted_over_supported():
    """Conservative merge policy: SUPPORTED+CONTRADICTED keeps CONTRADICTED."""
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    a = build_node(label="Alpha", definition="a", type="entity")
    b = build_node(label="Beta", definition="b", type="entity")
    c = build_node(label="Gamma", definition="c", type="entity")
    for n in (a, b, c):
        store.nodes[n.node_id] = n
    # Lex-first original id is SUPPORTED so a keep-wins policy would drop CONTRADICTED.
    e_sup = build_edge(
        source_node_ids=[a.node_id], target_node_ids=[c.node_id],
        direction="causal", relation_type="increases",
    ).model_copy(update={
        "edge_id": "aaa-supported", "status": EdgeStatus.SUPPORTED, "confidence": 0.9,
    })
    e_con = build_edge(
        source_node_ids=[b.node_id], target_node_ids=[c.node_id],
        direction="causal", relation_type="increases",
    ).model_copy(update={
        "edge_id": "zzz-contradicted", "status": EdgeStatus.CONTRADICTED, "confidence": 0.2,
        "open_risks": ["conflict"],
    })
    store.edges = {e_sup.edge_id: e_sup, e_con.edge_id: e_con}
    delta = _extract([MergeNodesOp(survivor_node_id=a.node_id, merged_node_id=b.node_id)],
                     store.base_hash)
    applied = v.apply(store, delta)
    survived = next(iter(applied.edges.values()))
    assert survived.status == EdgeStatus.CONTRADICTED
    assert survived.confidence == 0.2
    assert "conflict" in survived.open_risks


def test_merge_remaps_experiment_plans_and_evidence_link_target_ids():
    from src.graph_store import EvidenceLink, EvidenceRole, compute_target_id

    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    a = build_node(label="Alpha", definition="a", type="entity")
    b = build_node(label="Beta", definition="b", type="entity")
    c = build_node(label="Gamma", definition="c", type="entity")
    for n in (a, b, c):
        store.nodes[n.node_id] = n
    edge = build_edge(
        source_node_ids=[b.node_id], target_node_ids=[c.node_id],
        direction="causal", relation_type="increases",
    )
    store.edges[edge.edge_id] = edge
    store.experiment_plans[edge.edge_id] = ExperimentPlan(
        hypothesis_under_test="B causes C", design=ExperimentDesign.ABLATION,
    )
    old_target = compute_target_id(edge.edge_id, EvidenceRole.SUPPORT)
    store.evidence_links = [
        EvidenceLink(
            target_id=old_target,
            verification_task_id="vt-1",
            evidence_id="E1",
            evidence_role=EvidenceRole.SUPPORT,
            retrieval_event_id="r1",
            committed_transaction_id="tx-1",
            trust_tier="A",
        )
    ]
    old_id = edge.edge_id
    applied = v.apply(
        store,
        _extract([MergeNodesOp(survivor_node_id=a.node_id, merged_node_id=b.node_id)],
                 store.base_hash),
    )
    new_id = next(iter(applied.edges))
    assert old_id not in applied.edges
    assert old_id not in applied.experiment_plans
    assert new_id in applied.experiment_plans
    assert applied.evidence_links[0].target_id == compute_target_id(
        new_id, EvidenceRole.SUPPORT
    )


def test_forged_edge_id_fails_schema_gate():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    n1 = build_node(label="x", type="construct")
    n2 = build_node(label="y", type="outcome")
    store.nodes[n1.node_id] = n1
    store.nodes[n2.node_id] = n2
    forged = build_edge(
        source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
        direction="causal", relation_type="increases",
    ).model_copy(update={"edge_id": "FORGED"})
    delta = _extract([AddEdgeOp(edge=forged)], store.base_hash)
    decision = v.evaluate(store, delta)
    assert decision.schema is False
    assert decision.failing_gate == "schema"


def test_forged_node_id_fails_schema_gate():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    forged = build_node(label="z", type="construct").model_copy(update={"node_id": "FORGED-NODE"})
    delta = _extract([AddNodeOp(node=forged)], store.base_hash)
    decision = v.evaluate(store, delta)
    assert decision.schema is False
    assert decision.failing_gate == "schema"


# --- Δ^hypothesis routes through the Operation union ------------------------------------
def test_hypothesis_applies_via_the_operation_union_landing_unverified():
    v = GraphDeltaValidator()
    store = CausalClaimGraphStore()
    n1 = build_node(label="chronic stress", type="exposure/intervention")
    n2 = build_node(label="hypertension", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")

    hypothesis = GraphDeltaProposal(
        family=DeltaFamily.HYPOTHESIS, base_graph_hash=store.base_hash,
        payload=HypothesisPayload(new_nodes=[n1, n2], new_edges=[edge]),
    )
    assert v.evaluate(store, hypothesis).accept_i == 1
    applied = v.apply(store, hypothesis)
    assert applied.edges[edge.edge_id].status == EdgeStatus.UNVERIFIED

    # apply mutates ONLY via the Operation union: the hypothesis (new_nodes/new_edges) is
    # sugar, so its result graph is byte-identical to the equivalent add_node/add_edge extract.
    extract = _extract(
        [AddNodeOp(node=n1), AddNodeOp(node=n2), AddEdgeOp(edge=edge)], store.base_hash
    )
    assert v.apply(store, extract).content_hash() == applied.content_hash()


# --- Experiment deltas ----------------------------------------------------------------
def _experiment_delta(store, hypothesis_id, *, evidence_ids=("E1",)) -> GraphDeltaProposal:
    plan = ExperimentPlan(
        hypothesis_under_test=f"edge {hypothesis_id}",
        design=ExperimentDesign.RANDOMIZED_CONTROLLED,
    )
    return GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT, base_graph_hash=store.base_hash,
        payload=ExperimentPayload(
            hypothesis_id=hypothesis_id, experiment_plan=plan,
            referenced_evidence_ids=list(evidence_ids),
        ),
    )


def test_experiment_commits_through_all_five_gates_and_stores_the_plan():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)  # a committed hypothesis edge
    delta = _experiment_delta(store, edge.edge_id, evidence_ids=("E1",))
    assert v.evaluate(store, delta).accept_i == 1
    applied = v.apply(store, delta)
    assert applied.version == store.version + 1                     # advances one version (v5)
    assert edge.edge_id in applied.experiment_plans                 # plan stored, keyed by edge id
    # The bound edge's status and confidence remain unchanged because experiments carry no verdict.
    assert applied.edges[edge.edge_id].status == EdgeStatus.SUPPORTED
    assert applied.edges[edge.edge_id].confidence == store.edges[edge.edge_id].confidence


def test_experiment_unresolved_hypothesis_id_fails_refs():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    store, _edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    delta = _experiment_delta(store, "NOT_A_COMMITTED_EDGE", evidence_ids=("E1",))
    decision = v.evaluate(store, delta)
    assert decision.refs is False
    assert decision.accept_i == 0


def test_experiment_unresolved_evidence_id_fails_refs():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})  # ledger has E1 only
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    delta = _experiment_delta(store, edge.edge_id, evidence_ids=("E1", "E_MISSING"))
    assert v.evaluate(store, delta).refs is False


def test_experiment_never_changes_edge_status_transition_gate_passes():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    store, edge = _seed_one_edge(EdgeStatus.UNVERIFIED)
    delta = _experiment_delta(store, edge.edge_id)
    # Experiments carry no verdict, so the transition gate always passes as it does for priority deltas.
    assert v.evaluate(store, delta).allowed_transition is True


def test_experiment_idempotency_gate_rejects_recommit():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    delta = _experiment_delta(store, edge.edge_id)
    assert v.evaluate(store, delta, committed_keys=set()).idempotent is True
    assert v.evaluate(store, delta, committed_keys={delta.idempotency_key()}).idempotent is False


def test_experiment_wrong_payload_type_fails_the_schema_gate():
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    store, edge = _seed_one_edge(EdgeStatus.SUPPORTED)
    delta = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT, base_graph_hash=store.base_hash,
        payload=PriorityPayload(node_or_edge_ids=[edge.edge_id], priority_values=[0.5]),
    )
    decision = v.evaluate(store, delta)
    assert decision.schema is False
    assert decision.failing_gate == "schema"
