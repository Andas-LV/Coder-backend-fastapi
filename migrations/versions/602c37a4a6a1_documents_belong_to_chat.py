"""documents belong to chat

Files are no longer a separate library: every document is attached to exactly one chat
(through a message) and is deleted together with it. The many-to-many table
chat_documents is replaced by documents.chat_id.

Revision ID: 602c37a4a6a1
Revises: 08361227c658
Create Date: 2026-11-09 17:20:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "602c37a4a6a1"
down_revision: str | Sequence[str] | None = "08361227c658"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("chat_id", sa.Uuid(), nullable=True))

    # Carry over existing links. A document that was shared by several chats keeps only
    # one of them (the one with the smallest id).
    op.execute(
        """
        UPDATE documents d
        SET chat_id = (
            SELECT cd.chat_id FROM chat_documents cd
            WHERE cd.document_id = d.id
            ORDER BY cd.chat_id
            LIMIT 1
        )
        """
    )
    # Library documents that were never attached to a chat are unreachable now (there is
    # no documents API any more). Their chunks go with them (ON DELETE CASCADE).
    op.execute("DELETE FROM documents WHERE chat_id IS NULL")

    op.alter_column("documents", "chat_id", nullable=False)
    op.create_foreign_key(
        "documents_chat_id_fkey",
        "documents",
        "chats",
        ["chat_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(op.f("ix_documents_chat_id"), "documents", ["chat_id"])
    op.drop_table("chat_documents")


def downgrade() -> None:
    op.create_table(
        "chat_documents",
        sa.Column("chat_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chat_id", "document_id"),
    )
    op.execute(
        "INSERT INTO chat_documents (chat_id, document_id) SELECT chat_id, id FROM documents"
    )
    op.drop_index(op.f("ix_documents_chat_id"), table_name="documents")
    op.drop_constraint("documents_chat_id_fkey", "documents", type_="foreignkey")
    op.drop_column("documents", "chat_id")
