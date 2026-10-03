"""Persist dataset Network Health runs separately from manual predictions."""

import sqlalchemy as sa
from alembic import op

revision = "0004_network_health"
down_revision = "0003_authentication"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "network_health_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dataset_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("started_at", sa.String(40), nullable=False),
        sa.Column("completed_at", sa.String(40)),
        sa.Column("model_version", sa.String(128)),
        sa.Column("record_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary_json", sa.Text()),
        sa.Column("error", sa.Text()),
    )
    op.create_table(
        "network_health_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("network_health_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cell", sa.String(128), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("ground_truth_cause", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("predicted_cause", sa.String(16)),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_network_health_results_run_status", "network_health_results", ["run_id", "status"]
    )
    op.create_index(
        "ix_network_health_results_run_cell", "network_health_results", ["run_id", "cell"]
    )


def downgrade() -> None:
    op.drop_index("ix_network_health_results_run_cell", table_name="network_health_results")
    op.drop_index("ix_network_health_results_run_status", table_name="network_health_results")
    op.drop_table("network_health_results")
    op.drop_table("network_health_runs")
