#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations

# required for typing
from typing import Any
from itertools import combinations

# required for development
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAWI, OneShotSyncAgent
from scml.oneshot.agents.rand import distribute

# required for typing
from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState


class SupMerKos(OneShotSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `OneShotUFun` in the docs for more details).
    """

    def init(self):
        """Called once after the agent-world interface is initialized"""
        self._best_partner_selling: dict[str, int] = {}
        self._best_partner_buying: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        self._best_step_selling: dict[str, int] = {}
        self._best_step_buying: dict[str, int] = {}

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        return {
            partner: self._make_offer(partner, quantity, t=0.0)
            for partner, quantity in self._distribute_needs(t=0.0).items()
        }

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        responses: dict[str, SAOResponse] = {}
        current_offers = {
            p: o
            for p, o in offers.items()
            if o is not None and o[TIME] == self.awi.current_step
        }
        future_partners = set(offers).difference(current_offers)
        t = min((s.relative_time for s in states.values()), default=1.0)

        for partner, offer in current_offers.items():
            self._record_offer(partner, offer)

        accepted = self._best_acceptable_subset(current_offers, t)
        accepted_qty = sum(current_offers[p][QUANTITY] for p in accepted)
        for partner in accepted:
            responses[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER, current_offers[partner]
            )

        remaining_partners = [
            p for p in current_offers if p not in accepted
        ] + list(future_partners)
        remaining_need = max(0, self._needed() - accepted_qty)
        distribution = self._distribute_quantity(remaining_need, remaining_partners, t)

        for partner in remaining_partners:
            quantity = distribution.get(partner, 0)
            offer = self._make_offer(partner, quantity, t)
            if offer is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        return responses

    def _best_acceptable_subset(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if not offers:
            return set()

        partners = list(offers)
        baseline = self._utility({})
        best_partners: tuple[str, ...] = tuple()
        best_utility = baseline
        best_mismatch = float("inf")
        need = self._needed()

        for n in range(1, len(partners) + 1):
            for selected in combinations(partners, n):
                selected_offers = {p: offers[p] for p in selected}
                if not all(self._price_is_acceptable(p, offers[p], t) for p in selected):
                    continue
                utility = self._utility(selected_offers)
                mismatch = abs(sum(offers[p][QUANTITY] for p in selected) - need)
                if utility > best_utility + 1e-9 or (
                    abs(utility - best_utility) <= 1e-9 and mismatch < best_mismatch
                ):
                    best_partners = selected
                    best_utility = utility
                    best_mismatch = mismatch

        if not best_partners:
            return set()

        allowed_mismatch = self._allowed_mismatch(t)
        min_gain = max(0.0, (1.0 - t) * 0.02 * max(1, self.awi.n_lines))
        if best_utility < baseline + min_gain:
            return set()
        if best_mismatch > allowed_mismatch:
            return set()
        return set(best_partners)

    def _utility(self, offers: dict[str, Outcome]) -> float:
        return float(self.ufun.from_offers(offers, ignore_signed_contracts=False))

    def _make_offer(self, partner: str, quantity: int, t: float) -> Outcome | None:
        nmi = self.get_nmi(partner)
        if nmi is None or quantity <= 0:
            return None
        q_issue = nmi.issues[QUANTITY]
        quantity = max(q_issue.min_value, min(int(quantity), q_issue.max_value))
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, self._target_price(partner, t))

    def _target_price(self, partner: str, t: float) -> int:
        nmi = self.get_nmi(partner)
        issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = int(issue.min_value), int(issue.max_value)
        reservation = max(pmin, min(pmax, self._reservation_price()))
        concession = min(1.0, max(0.0, t)) ** 1.8

        if self._is_selling_partner(partner):
            target = pmax - (pmax - reservation) * concession
            best_seen = self._best_step_selling.get(
                partner, self._best_partner_selling.get(partner)
            )
            if best_seen is not None:
                target = max(reservation, 0.75 * target + 0.25 * best_seen)
        else:
            target = pmin + (reservation - pmin) * concession
            best_seen = self._best_step_buying.get(
                partner, self._best_partner_buying.get(partner)
            )
            if best_seen is not None:
                target = min(reservation, 0.75 * target + 0.25 * best_seen)

        return max(pmin, min(pmax, int(round(target))))

    def _price_is_acceptable(self, partner: str, offer: Outcome, t: float) -> bool:
        price = offer[UNIT_PRICE]
        threshold = self._target_price(partner, min(1.0, t + 0.12))
        if self._is_selling_partner(partner):
            return price >= threshold
        return price <= threshold

    def _reservation_price(self) -> int:
        production_cost = int(round(getattr(self.awi.profile, "cost", 0) or 0))
        if self.awi.is_first_level:
            q = self.awi.current_exogenous_input_quantity
            unit_input = (
                self.awi.current_exogenous_input_price / q
                if q
                else self.awi.trading_prices[self.awi.my_input_product]
            )
            buffer = 0.15 * max(
                self.awi.current_disposal_cost, self.awi.current_shortfall_penalty
            )
            return int(round(unit_input + production_cost + buffer))

        q = self.awi.current_exogenous_output_quantity
        unit_output = (
            self.awi.current_exogenous_output_price / q
            if q
            else self.awi.trading_prices[self.awi.my_output_product]
        )
        buffer = 0.15 * max(
            self.awi.current_disposal_cost, self.awi.current_shortfall_penalty
        )
        return int(round(unit_output - production_cost - buffer))

    def _needed(self) -> int:
        return int(self.awi.needed_sales if self.awi.is_first_level else self.awi.needed_supplies)

    def _distribute_needs(self, t: float) -> dict[str, int]:
        partners = list(self.active_negotiators.keys())
        need = self._needed()
        overorder = 1.15 - 0.15 * (min(1.0, max(0.0, t)) ** 0.7)
        return self._distribute_quantity(int(round(need * overorder)), partners, t)

    def _distribute_quantity(
        self, quantity: int, partners: list[str], t: float
    ) -> dict[str, int]:
        partners = [p for p in partners if p in self.negotiators]
        if not partners or quantity <= 0:
            return {p: 0 for p in partners}
        mx = max(
            self.get_nmi(p).issues[QUANTITY].max_value
            for p in partners
            if self.get_nmi(p) is not None
        )
        concentrated = t < 0.55
        quantities = distribute(
            quantity,
            len(partners),
            mx=mx,
            equal=not concentrated,
            concentrated=concentrated,
            allow_zero=self.awi.allow_zero_quantity,
        )
        return dict(zip(partners, quantities, strict=False))

    def _allowed_mismatch(self, t: float) -> float:
        return max(0.0, self.awi.n_lines * (0.08 + 0.55 * (t**2.5)))

    def _is_selling_partner(self, partner: str) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return self.awi.is_first_level
        return nmi.annotation["product"] == self.awi.my_output_product

    def _record_offer(self, partner: str, offer: Outcome) -> None:
        price = int(offer[UNIT_PRICE])
        if self._is_selling_partner(partner):
            self._best_step_selling[partner] = max(
                price, self._best_step_selling.get(partner, price)
            )
            self._best_partner_selling[partner] = max(
                price, self._best_partner_selling.get(partner, price)
            )
        else:
            self._best_step_buying[partner] = min(
                price, self._best_step_buying.get(partner, price)
            )
            self._best_partner_buying[partner] = min(
                price, self._best_partner_buying.get(partner, price)
            )

    # =====================
    # Time-Driven Callbacks
    # =====================

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""
        for partner in partners:
            self._failures[partner] = self._failures.get(partner, 0) + 1

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            price = contract.agreement["unit_price"]
            self._best_partner_selling[partner] = max(
                price, self._best_partner_selling.get(partner, price)
            )
        else:
            partner = contract.annotation["seller"]
            price = contract.agreement["unit_price"]
            self._best_partner_buying[partner] = min(
                price, self._best_partner_buying.get(partner, price)
            )
        self._successes[partner] = self._successes.get(partner, 0) + 1


if __name__ == "__main__":
    import sys

    from myagent.helpers.runner import run

    run([SupMerKos], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
