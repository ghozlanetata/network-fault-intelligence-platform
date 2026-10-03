"""Create prediction history.

Revision ID: 0001_prediction_history
Revises:
Create Date: 2026-09-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_prediction_history"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cell", sa.String(length=128), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("fault", sa.Boolean(), nullable=False),
        sa.Column("cause", sa.String(length=16), nullable=False),
        sa.Column("event_date", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("kpis_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_predictions_cell_id", "predictions", ["cell", "id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_cell_id", table_name="predictions")
    op.drop_table("predictions")
