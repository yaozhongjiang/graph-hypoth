"""Deterministic five-gate commit boundary and pure delta application.

Acceptance requires valid schema, resolvable references, a current base hash, an allowed status
transition, and an unused idempotency key. ``apply`` returns a new store and never mutates its
input, so rejected proposals cannot partially change graph state. Receipt orchestration lives in
``transaction_log.py``.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

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
    VERDICT_TO_STATUS,
    VerifyPayload,
    compute_edge_id,
    compute_node_id,
)
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    EdgeStatus,
    ExperimentPlan,
    compute_target_id,
)

_FAMILY_PAYLOAD = {
    DeltaFamily.EXTRACT: ExtractPayload,
    DeltaFamily.PRIORITY: PriorityPayload,
    DeltaFamily.VERIFY: VerifyPayload,
    DeltaFamily.HYPOTHESIS: HypothesisPayload,
    DeltaFamily.EXPERIMENT: ExperimentPayload,
}

# Per-family allowed verification transitions. Extraction and hypothesis edges begin
# `unverified` only; priority changes no status. Settled states are terminal EXCEPT
# `insufficient`, which may move to a non-insufficient settled status on new evidence;
# verify never returns a target to `unverified`.
_SETTLED_NON_INSUFFICIENT = frozenset(
    {EdgeStatus.SUPPORTED, EdgeStatus.CONTRADICTED, EdgeStatus.QUALIFIED, EdgeStatus.NOT_CAUSAL}
)
_ALLOWED_VERIFY_TRANSITIONS: dict[EdgeStatus, frozenset[EdgeStatus]] = {
    EdgeStatus.UNVERIFIED: _SETTLED_NON_INSUFFICIENT | {EdgeStatus.INSUFFICIENT},
    EdgeStatus.INSUFFICIENT: _SETTLED_NON_INSUFFICIENT,
    EdgeStatus.SUPPORTED: frozenset(),
    EdgeStatus.CONTRADICTED: frozenset(),
    EdgeStatus.QUALIFIED: frozenset(),
    EdgeStatus.NOT_CAUSAL: frozenset(),
}

# Higher rank wins on merge collision: prefer the more conservative settled verdict so a
# SUPPORTED+CONTRADICTED collapse cannot erase the contradiction.
_STATUS_CONSERVATISM: dict[EdgeStatus, int] = {
    EdgeStatus.UNVERIFIED: 0,
    EdgeStatus.SUPPORTED: 1,
    EdgeStatus.INSUFFICIENT: 2,
    EdgeStatus.QUALIFIED: 3,
    EdgeStatus.NOT_CAUSAL: 4,
    EdgeStatus.CONTRADICTED: 5,
}

_GATE_ORDER = ("schema", "refs", "base_hash", "allowed_transition", "idempotent")


@dataclass(frozen=True)
class GateResults:
    """The five gate decisions; acceptance requires all of them."""

    schema: bool
    refs: bool
    base_hash: bool
    allowed_transition: bool
    idempotent: bool

    @property
    def accept_i(self) -> int:
        return int(
            self.schema
            and self.refs
            and self.base_hash
            and self.allowed_transition
            and self.idempotent
        )

    @property
    def accepted(self) -> bool:
        return self.accept_i == 1

    @property
    def failing_gate(self) -> str | None:
        for name in _GATE_ORDER:
            if getattr(self, name) is False:
                return name
        return None


class GraphDeltaValidator:
    """Own commit authority through five gates and deterministic application."""

    def __init__(self, ledger_evidence_ids: Collection[str] = ()) -> None:
        self._ledger = frozenset(ledger_evidence_ids)

    # --- Accept_i ----------------------------------------------------------------------
    def evaluate(
        self,
        store: CausalClaimGraphStore,
        delta: GraphDeltaProposal,
        committed_keys: Collection[str] = frozenset(),
    ) -> GateResults:
        schema = self._gate_schema(delta)
        base_hash = delta.base_graph_hash == store.base_hash
        idempotent = delta.idempotency_key() not in set(committed_keys)
        # refs/transition read the delta payload, so they only run on a schema-valid delta.
        refs = self._gate_refs(store, delta) if schema else False
        transition = self._gate_allowed_transition(store, delta) if schema else False
        return GateResults(
            schema=schema, refs=refs, base_hash=base_hash,
            allowed_transition=transition, idempotent=idempotent,
        )

    def _gate_schema(self, delta: GraphDeltaProposal) -> bool:
        """Return whether payload shape and constrained values match the declared family.

        For a priority delta, 1_schema also enforces the ``[0,1]`` priority scale so a raw
        ``PriorityPayload`` that bypasses the authoring annotation is still rejected, never
        applied. Extract and hypothesis operations must carry content-addressed node/edge
        ids (``compute_node_id`` / ``compute_edge_id``).
        """
        payload = delta.payload
        if not isinstance(payload, _FAMILY_PAYLOAD[delta.family]):
            return False
        if isinstance(payload, PriorityPayload):
            # 1:1 pairing: mismatched id/value lengths would let apply()'s zip() silently
            # commit only the zipped prefix, so reject it at the schema gate.
            if len(payload.node_or_edge_ids) != len(payload.priority_values):
                return False
            return all(0.0 <= value <= 1.0 for value in payload.priority_values)
        operations = _operations_of(payload)
        if operations is not None:
            return _content_addressed_operations(operations)
        return True

    def _gate_refs(self, store: CausalClaimGraphStore, delta: GraphDeltaProposal) -> bool:
        """Return whether every node, edge, and ledger reference resolves."""
        payload = delta.payload
        operations = _operations_of(payload)
        if operations is not None:  # extract + hypothesis both flow through the Operation union
            available = set(store.nodes) | {
                op.node.node_id for op in operations if isinstance(op, AddNodeOp)
            }
            for op in operations:
                if isinstance(op, AddEdgeOp):
                    endpoints = set(op.edge.source_node_ids) | set(op.edge.target_node_ids)
                    if not endpoints <= available:
                        return False
                elif isinstance(op, MergeNodesOp):
                    if op.survivor_node_id not in available or op.merged_node_id not in available:
                        return False
            return True
        if isinstance(payload, VerifyPayload):
            if payload.edge_id not in store.edges:
                return False
            return all(link.evidence_id in self._ledger for link in payload.evidence_links)
        if isinstance(payload, PriorityPayload):
            # Priority is node-only because ``user_priority`` lives on ConceptNode.
            return all(ref in store.nodes for ref in payload.node_or_edge_ids)
        if isinstance(payload, ExperimentPayload):
            # The plan binds to a committed hypothesis edge, and every grounding evidence ID
            # resolves in the ledger.
            if payload.hypothesis_id not in store.edges:
                return False
            return all(eid in self._ledger for eid in payload.referenced_evidence_ids)
        return False

    def _gate_allowed_transition(
        self, store: CausalClaimGraphStore, delta: GraphDeltaProposal
    ) -> bool:
        """Return whether the requested status change is legal for the family."""
        payload = delta.payload
        operations = _operations_of(payload)
        if operations is not None:  # extract + hypothesis: every new edge lands `unverified`
            return all(
                op.edge.status == EdgeStatus.UNVERIFIED
                for op in operations
                if isinstance(op, AddEdgeOp)
            )
        if isinstance(payload, VerifyPayload):
            edge = store.edges.get(payload.edge_id)
            if edge is None:
                return False
            target_status = VERDICT_TO_STATUS[payload.verdict]
            return target_status in _ALLOWED_VERIFY_TRANSITIONS[edge.status]
        if isinstance(payload, PriorityPayload):
            return True  # Priority carries no status.
        if isinstance(payload, ExperimentPayload):
            return True  # Experiment attaches a plan and carries no verdict.
        return False

    # --- apply -------------------------------------------------------------------------
    def apply(
        self,
        store: CausalClaimGraphStore,
        delta: GraphDeltaProposal,
        *,
        transaction_id: str = "tx-uncommitted",
    ) -> CausalClaimGraphStore:
        """Apply a delta to a copy of the store and advance its version.

        ``transaction_id`` stamps committed evidence links; commit and replay pass the
        deterministic id so the result hash is replay-stable.
        """
        new = store.model_copy(deep=True)
        new.version = store.version + 1
        payload = delta.payload
        operations = _operations_of(payload)
        if operations is not None:  # extract + hypothesis mutate ONLY via the Operation union
            self._apply_operations(new, operations)
        elif isinstance(payload, VerifyPayload):
            edge = new.edges[payload.edge_id]
            edge.status = VERDICT_TO_STATUS[payload.verdict]
            edge.confidence = payload.confidence
            edge.open_risks = sorted(set(edge.open_risks) | set(payload.open_risks))
            for link in payload.evidence_links:
                new.evidence_links.append(
                    link.model_copy(update={"committed_transaction_id": transaction_id})
                )
        elif isinstance(payload, PriorityPayload):
            for ref, value in zip(payload.node_or_edge_ids, payload.priority_values):
                if ref in new.nodes:
                    new.nodes[ref].user_priority = value
        elif isinstance(payload, ExperimentPayload):
            # Bind the validated plan to its hypothesis edge without changing status or confidence.
            new.experiment_plans[payload.hypothesis_id] = payload.experiment_plan.model_copy(
                deep=True
            )
        return new

    def _apply_operations(self, store: CausalClaimGraphStore, operations) -> None:
        for op in operations:
            if isinstance(op, AddNodeOp):
                store.nodes.setdefault(op.node.node_id, op.node)  # re-add identical = no-op
            elif isinstance(op, AddEdgeOp):
                landed = op.edge.model_copy(update={"status": EdgeStatus.UNVERIFIED})
                store.edges.setdefault(landed.edge_id, landed)
            elif isinstance(op, MergeNodesOp):
                self._merge_nodes(store, op.survivor_node_id, op.merged_node_id)

    def _merge_nodes(
        self, store: CausalClaimGraphStore, survivor_id: str, merged_id: str
    ) -> None:
        """Redirect edges, absorb the survivor, retire the merged node, and deduplicate."""
        if survivor_id not in store.nodes or merged_id not in store.nodes:
            return  # 1_refs guards this; defensive no-op
        survivor = store.nodes[survivor_id]
        merged = store.nodes[merged_id]

        # 2. survivor absorbs aliases ∪ {merged.label}, unions scope_qualifiers/provenance.
        survivor.aliases = sorted(set(survivor.aliases) | set(merged.aliases) | {merged.label})
        survivor.scope_qualifiers = sorted(
            set(survivor.scope_qualifiers) | set(merged.scope_qualifiers)
        )
        survivor.provenance = (
            survivor.provenance
            + merged.provenance
            + [{"merged_from": merged_id, "merged_label": merged.label}]
        )

        # 1 + 4. redirect every edge off merged_id, recompute its content-addressed id, and
        # de-dup colliding ids. Iterating in ascending original-edge_id order makes the
        # first survivor of a collision the lexicographically smaller one before the
        # conservative status policy in ``_union_edge_lists`` may prefer the other scalars.
        redirected: dict[str, CausalEdge] = {}
        id_map: dict[str, str] = {}
        for original_id in sorted(store.edges):
            edge = store.edges[original_id]
            src = [survivor_id if x == merged_id else x for x in edge.source_node_ids]
            tgt = [survivor_id if x == merged_id else x for x in edge.target_node_ids]
            new_id = compute_edge_id(src, tgt, edge.direction, edge.relation_type)
            moved = edge.model_copy(
                update={"edge_id": new_id, "source_node_ids": src, "target_node_ids": tgt}
            )
            id_map[original_id] = new_id
            if new_id in redirected:
                redirected[new_id] = _union_edge_lists(redirected[new_id], moved)
            else:
                redirected[new_id] = moved
        store.edges = redirected
        _remap_edge_dependents(store, id_map)

        # 3. retire merged node (lineage preserved in survivor.provenance).
        del store.nodes[merged_id]


def _operations_of(payload) -> list[AddNodeOp | AddEdgeOp | MergeNodesOp] | None:
    """The ``Operation`` list for the union-based families — ``Δ^extract`` directly,
    ``Δ^hypothesis`` desugared, or ``None`` for verification and priority. This keeps
    ``apply`` mutations within the closed operation union."""
    if isinstance(payload, ExtractPayload):
        return payload.operations
    if isinstance(payload, HypothesisPayload):
        return payload.to_operations()
    return None


def _content_addressed_operations(operations: list[AddNodeOp | AddEdgeOp | MergeNodesOp]) -> bool:
    """Reject forged node/edge ids that are not the content-addressed hash of their fields."""
    for op in operations:
        if isinstance(op, AddNodeOp):
            node = op.node
            if node.node_id != compute_node_id(node.type, node.label, node.definition):
                return False
        elif isinstance(op, AddEdgeOp):
            edge = op.edge
            if edge.edge_id != compute_edge_id(
                edge.source_node_ids, edge.target_node_ids, edge.direction, edge.relation_type
            ):
                return False
    return True


def _remap_edge_dependents(store: CausalClaimGraphStore, id_map: dict[str, str]) -> None:
    """Rewrite ``experiment_plans`` keys and ``evidence_links.target_id`` after edge-id remap.

    Plan collisions (two old edges collapse to one id) keep the plan from the
    lexicographically smaller original edge id. Links whose ``target_id`` matches
    ``compute_target_id(old_id, role)`` are rewritten to the new edge id.
    """
    if any(old != new for old, new in id_map.items()):
        remapped_plans: dict[str, ExperimentPlan] = {}
        claimed_by: dict[str, str] = {}
        for old_id in sorted(store.experiment_plans):
            plan = store.experiment_plans[old_id]
            new_id = id_map.get(old_id, old_id)
            prior = claimed_by.get(new_id)
            if prior is not None and prior < old_id:
                continue  # keep plan from the smaller original key
            remapped_plans[new_id] = plan
            claimed_by[new_id] = old_id
        store.experiment_plans = remapped_plans

        new_links = []
        for link in store.evidence_links:
            updated = link
            for old_id, new_id in id_map.items():
                if old_id == new_id:
                    continue
                if link.target_id == compute_target_id(old_id, link.evidence_role):
                    updated = link.model_copy(
                        update={"target_id": compute_target_id(new_id, link.evidence_role)}
                    )
                    break
            new_links.append(updated)
        store.evidence_links = new_links


def _more_conservative_edge(left: CausalEdge, right: CausalEdge) -> CausalEdge:
    """Return the edge whose status/confidence should win a merge collision."""
    left_rank = _STATUS_CONSERVATISM[left.status]
    right_rank = _STATUS_CONSERVATISM[right.status]
    if right_rank > left_rank:
        return right
    if left_rank > right_rank:
        return left
    # Same status: prefer the lower confidence (more conservative); None loses to a number.
    left_conf = left.confidence
    right_conf = right.confidence
    if left_conf is None and right_conf is not None:
        return right
    if right_conf is None and left_conf is not None:
        return left
    if left_conf is not None and right_conf is not None and right_conf < left_conf:
        return right
    return left


def _union_edge_lists(keep: CausalEdge, other: CausalEdge) -> CausalEdge:
    """Merge two edges that collapsed to one canonical id.

    List-valued fields are unioned. Scalar fields (status, confidence, mechanism) come from
    the more conservative edge so a SUPPORTED+CONTRADICTED collision cannot drop the
    contradiction. ``CausalEdge`` carries no provenance field, so edge provenance is not
    merged here.
    """
    winner = _more_conservative_edge(keep, other)
    return winner.model_copy(
        update={
            "open_risks": sorted(set(keep.open_risks) | set(other.open_risks)),
            "confounders": sorted(set(keep.confounders) | set(other.confounders)),
            "conditions": sorted(set(keep.conditions) | set(other.conditions)),
        }
    )
