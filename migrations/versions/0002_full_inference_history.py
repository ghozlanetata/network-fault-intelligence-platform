"""Persist full inference results and allow optional site context.

Revision ID: 0002_full_inference_history
Revises: 0001_prediction_history
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_full_inference_history"
down_revision = "0001_prediction_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        for name, kind in (
            ("cell", sa.String(length=128)),
            ("latitude", sa.Float()),
            ("longitude", sa.Float()),
            ("fault", sa.Boolean()),
            ("cause", sa.String(length=16)),
            ("event_date", sa.String(length=10)),
        ):
            batch.alter_column(name, existing_type=kind, nullable=True)
        batch.add_column(
            sa.Column("status", sa.String(length=24), nullable=False, server_default="NORMAL")
        )
        batch.add_column(
            sa.Column(
                "model_version",
                sa.String(length=128),
                nullable=False,
                server_default="legacy-unknown",
            )
        )
        batch.add_column(sa.Column("result_json", sa.Text(), nullable=True))

    op.execute(
        "UPDATE predictions SET status = CASE WHEN fault = 1 THEN 'FAULT' ELSE 'NORMAL' END"
    )


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_column("result_json")
        batch.drop_column("model_version")
        batch.drop_column("status")
        for name, kind in (
            ("cell", sa.String(length=128)),
            ("latitude", sa.Float()),
            ("longitude", sa.Float()),
            ("fault", sa.Boolean()),
            ("cause", sa.String(length=16)),
            ("event_date", sa.String(length=10)),
        ):
            batch.alter_column(name, existing_type=kind, nullable=False)
