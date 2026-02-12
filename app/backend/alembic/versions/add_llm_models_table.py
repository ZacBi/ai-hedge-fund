"""add_llm_models_table

Revision ID: a1b2c3d4e5f6
Revises: d5e78f9a1b2c
Create Date: 2025-02-12 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "d5e78f9a1b2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("provider", sa.String(length=80), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_llm_models_id"), "llm_models", ["id"], unique=False)
    op.create_index(op.f("ix_llm_models_model_name"), "llm_models", ["model_name"], unique=False)
    op.create_index(op.f("ix_llm_models_provider"), "llm_models", ["provider"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_llm_models_provider"), table_name="llm_models")
    op.drop_index(op.f("ix_llm_models_model_name"), table_name="llm_models")
    op.drop_index(op.f("ix_llm_models_id"), table_name="llm_models")
    op.drop_table("llm_models")
