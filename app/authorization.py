"""Reusable application-role dependencies for FastAPI routes."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.auth import Role, User

SESSION_COOKIE = "nfi_session"


def require_authenticated_user(request: Request) -> User:
    user = request.app.state.auth.user_for_session(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "Authentication is required."},
        )
    return user


def require_any_role(*roles: Role) -> Callable[[User], User]:
    allowed = frozenset(Role(role) for role in roles)

    def dependency(
        user: Annotated[User, Depends(require_authenticated_user)],
    ) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=403,
                detail={"code": "forbidden", "message": "This role cannot access this resource."},
            )
        return user

    return dependency


def require_role(role: Role) -> Callable[[User], User]:
    return require_any_role(role)
