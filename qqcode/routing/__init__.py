"""Routing package: task classification and FastPath execution."""

from qqcode.routing.fastpath import FastPathInput, FastPathResult, execute_fastpath
from qqcode.routing.router import RoutingDecision, RoutingResult, route_task

__all__ = [
    "FastPathInput",
    "FastPathResult",
    "RoutingDecision",
    "RoutingResult",
    "execute_fastpath",
    "route_task",
]
