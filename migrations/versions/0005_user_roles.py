"""Add application roles to users.

Revision ID: 0005_user_roles
Revises: 0004_network_health
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_user_roles"
down_revision = "0004_network_health"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("role", sa.String(length=32), nullable=False, server_default="operator")
        )
        batch.create_check_constraint(
            "ck_users_role",
            "role IN ('operator', 'network_admin', 'platform_ml_admin')",
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("ck_users_role", type_="check")
        batch.drop_column("role")
