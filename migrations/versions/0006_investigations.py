"""Add investigation outcomes linked to existing predictions or dataset results."""

import sqlalchemy as sa
from alembic import op

revision = "0006_investigations"
down_revision = "0005_user_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "investigations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prediction_id", sa.Integer(), sa.ForeignKey("predictions.id")),
        sa.Column(
            "network_health_result_id",
            sa.Integer(),
            sa.ForeignKey("network_health_results.id"),
        ),
        sa.Column("cell", sa.String(128)),
        sa.Column("predicted_cause", sa.String(16)),
        sa.Column("investigation_status", sa.String(32), nullable=False),
        sa.Column("findings", sa.Text(), nullable=False, server_default=""),
        sa.Column("investigation_cause", sa.String(16)),
        sa.Column("resolution_notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("submitted_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
        sa.Column("submitted_at", sa.String(40)),
        sa.Column("verified_cause", sa.String(16)),
        sa.Column("verified_by", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("verified_at", sa.String(40)),
        sa.Column("verification_status", sa.String(24), nullable=False),
        sa.Column("resolution_status", sa.String(24), nullable=False),
        sa.Column("is_completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.CheckConstraint(
            "(prediction_id IS NOT NULL AND network_health_result_id IS NULL) OR "
            "(prediction_id IS NULL AND network_health_result_id IS NOT NULL)",
            name="ck_investigations_single_source",
        ),
        sa.CheckConstraint(
            "investigation_status IN ('DRAFT','IN_PROGRESS','PENDING_VERIFICATION','VERIFIED')",
            name="ck_investigations_status",
        ),
        sa.CheckConstraint(
            "verification_status IN ('PENDING','VERIFIED','UNRESOLVED','UNKNOWN')",
            name="ck_investigations_verification_status",
        ),
        sa.CheckConstraint(
            "resolution_status IN ('OPEN','RESOLVED','UNRESOLVED')",
            name="ck_investigations_resolution_status",
        ),
    )
    op.create_index("ix_investigations_submitted_by", "investigations", ["submitted_by"])
    op.create_index("ix_investigations_status", "investigations", ["investigation_status"])
    op.create_index(
        "uq_investigations_prediction", "investigations", ["prediction_id"], unique=True
    )
    op.create_index(
        "uq_investigations_network_result",
        "investigations",
        ["network_health_result_id"],
        unique=True,
    )
    op.create_table(
        "investigation_verification_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "investigation_id",
            sa.Integer(),
            sa.ForeignKey("investigations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("previous_status", sa.String(24)),
        sa.Column("previous_cause", sa.String(16)),
        sa.Column("new_status", sa.String(24), nullable=False),
        sa.Column("new_cause", sa.String(16)),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("verified_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("verified_at", sa.String(40), nullable=False),
    )
    op.create_index(
        "ix_investigation_verification_events_investigation",
        "investigation_verification_events",
        ["investigation_id", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_investigation_verification_events_investigation",
        table_name="investigation_verification_events",
    )
    op.drop_table("investigation_verification_events")
    op.drop_index("uq_investigations_network_result", table_name="investigations")
    op.drop_index("uq_investigations_prediction", table_name="investigations")
    op.drop_index("ix_investigations_status", table_name="investigations")
    op.drop_index("ix_investigations_submitted_by", table_name="investigations")
    op.drop_table("investigations")
