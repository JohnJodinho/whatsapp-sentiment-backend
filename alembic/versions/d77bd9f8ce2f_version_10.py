"""Version 10

Revision ID: d77bd9f8ce2f
Revises: 2bae362756b9
Create Date: 2025-11-02 04:03:43.796576

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd77bd9f8ce2f'
down_revision: Union[str, Sequence[str], None] = '2bae362756b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Step 1: Add columns as nullable
    op.add_column('messages', sa.Column('word_count', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('emojis_count', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('links_count', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('is_question', sa.Boolean(), nullable=True))
    op.add_column('messages', sa.Column('is_media', sa.Boolean(), nullable=True))
    
    # Step 2: Update existing rows with default values
    op.execute("""
        UPDATE messages 
        SET word_count = 0,
            emojis_count = 0,
            links_count = 0,
            is_question = FALSE,
            is_media = FALSE
        WHERE word_count IS NULL
    """)
    
    # Step 3: Make columns non-nullable
    op.alter_column('messages', 'word_count', nullable=False)
    op.alter_column('messages', 'emojis_count', nullable=False)
    op.alter_column('messages', 'links_count', nullable=False)
    op.alter_column('messages', 'is_question', nullable=False)
    op.alter_column('messages', 'is_media', nullable=False)

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('messages', 'is_media')
    op.drop_column('messages', 'is_question')
    op.drop_column('messages', 'links_count')
    op.drop_column('messages', 'emojis_count')
    op.drop_column('messages', 'word_count')
