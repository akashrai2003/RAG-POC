"""
Conversation service for managing chat history and messages using SQLite.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import uuid

from models.schemas import (
    Conversation, Message, ConversationWithMessages, 
    CreateConversationRequest, CreateMessageRequest,
    ConversationListResponse, ConversationStatsResponse,
    MessageRole
)
from config.settings import config


class ConversationService:
    """Service for managing conversations and messages in SQLite database."""
    
    def __init__(self, db_path: str = None):
        """Initialize the conversation service."""
        self.db_path = db_path or str(Path(config.sqlite_db_path).parent / "conversations.db")
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"✅ ConversationService initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    metadata TEXT,
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            # Create messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            
            conn.commit()
    
    def create_conversation(self, request: CreateConversationRequest) -> Conversation:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Insert conversation
            conn.execute("""
                INSERT INTO conversations (id, title, created_at, updated_at, user_id, metadata, message_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                request.title,
                now.isoformat(),
                now.isoformat(),
                request.user_id,
                json.dumps({}),
                0
            ))
            
            # Add initial message if provided
            if request.initial_message:
                self._add_message_to_db(
                    conn,
                    conversation_id,
                    MessageRole.USER,
                    request.initial_message,
                    {}
                )
                # Update message count
                conn.execute("UPDATE conversations SET message_count = 1 WHERE id = ?", (conversation_id,))
            
            conn.commit()
        
        return Conversation(
            id=conversation_id,
            title=request.title,
            created_at=now,
            updated_at=now,
            user_id=request.user_id,
            metadata={},
            message_count=1 if request.initial_message else 0
        )
    
    def get_conversation(self, conversation_id: str, include_messages: bool = False) -> Optional[ConversationWithMessages]:
        """Get a conversation by ID, optionally including messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get conversation
            cursor = conn.execute("""
                SELECT id, title, created_at, updated_at, user_id, metadata, message_count
                FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            conversation = Conversation(
                id=row['id'],
                title=row['title'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                user_id=row['user_id'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                message_count=row['message_count'] or 0
            )
            
            if include_messages:
                messages = self._get_conversation_messages(conn, conversation_id)
                return ConversationWithMessages(**conversation.dict(), messages=messages)
            
            return ConversationWithMessages(**conversation.dict(), messages=[])
    
    def _get_conversation_messages(self, conn: sqlite3.Connection, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation."""
        cursor = conn.execute("""
            SELECT id, conversation_id, role, content, metadata, timestamp
            FROM messages WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append(Message(
                id=row['id'],
                conversation_id=row['conversation_id'],
                role=row['role'],
                content=row['content'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else datetime.utcnow()
            ))
        
        return messages
    
    def add_message(self, request: CreateMessageRequest) -> Message:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Check if conversation exists
            cursor = conn.execute("SELECT id FROM conversations WHERE id = ?", (request.conversation_id,))
            if not cursor.fetchone():
                raise ValueError(f"Conversation {request.conversation_id} not found")
            
            # Add message
            self._add_message_to_db(
                conn,
                request.conversation_id,
                request.role,
                request.content,
                request.metadata or {}
            )
            
            # Update conversation updated_at and message count
            conn.execute("""
                UPDATE conversations 
                SET updated_at = ?, message_count = message_count + 1
                WHERE id = ?
            """, (now.isoformat(), request.conversation_id))
            
            conn.commit()
        
        return Message(
            id=message_id,
            conversation_id=request.conversation_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata or {},
            timestamp=now
        )
    
    def _add_message_to_db(self, conn: sqlite3.Connection, conversation_id: str, role: str, content: str, metadata: Dict[str, Any]) -> str:
        """Add a message to the database within an existing transaction."""
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        conn.execute("""
            INSERT INTO messages (id, conversation_id, role, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            conversation_id,
            role,
            content,
            json.dumps(metadata),
            now.isoformat()
        ))
        
        return message_id
    
    def list_conversations(
        self, 
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        order_by: str = "updated_at"
    ) -> ConversationListResponse:
        """List conversations with pagination."""
        offset = (page - 1) * page_size
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query
            where_clause = ""
            params = []
            
            if user_id:
                where_clause = "WHERE user_id = ?"
                params.append(user_id)
            
            # Get total count
            count_query = f"SELECT COUNT(*) as count FROM conversations {where_clause}"
            cursor = conn.execute(count_query, params)
            total_count = cursor.fetchone()['count']
            
            # Get conversations
            query = f"""
                SELECT id, title, created_at, updated_at, user_id, metadata, message_count
                FROM conversations {where_clause}
                ORDER BY {order_by} DESC
                LIMIT ? OFFSET ?
            """
            params.extend([page_size, offset])
            
            cursor = conn.execute(query, params)
            conversations = []
            
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row['id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                    user_id=row['user_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    message_count=row['message_count'] or 0
                ))
        
        has_more = offset + page_size < total_count
        
        return ConversationListResponse(
            conversations=conversations,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update a conversation's title."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE conversations 
                SET title = ?, updated_at = ?
                WHERE id = ?
            """, (title, datetime.utcnow().isoformat(), conversation_id))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def get_conversation_stats(self, user_id: Optional[str] = None) -> ConversationStatsResponse:
        """Get conversation statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build where clause
            where_clause = ""
            params = []
            
            if user_id:
                where_clause = "WHERE user_id = ?"
                params.append(user_id)
            
            # Get total conversations
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM conversations {where_clause}", params)
            total_conversations = cursor.fetchone()['count']
            
            # Get total messages
            message_query = f"""
                SELECT COUNT(*) as count FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                {where_clause}
            """
            cursor = conn.execute(message_query, params)
            total_messages = cursor.fetchone()['count']
            
            # Get conversations active in last 24 hours
            last_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            active_params = params + [last_24h]
            active_where = f"{where_clause} {'AND' if where_clause else 'WHERE'} updated_at >= ?"
            
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM conversations {active_where}", active_params)
            active_conversations_24h = cursor.fetchone()['count']
            
            # Calculate average messages per conversation
            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
            
            # Get most recent conversation
            recent_query = f"""
                SELECT MAX(updated_at) as recent FROM conversations {where_clause}
            """
            cursor = conn.execute(recent_query, params)
            recent_row = cursor.fetchone()
            most_recent = None
            if recent_row['recent']:
                most_recent = datetime.fromisoformat(recent_row['recent'])
        
        return ConversationStatsResponse(
            total_conversations=total_conversations,
            total_messages=total_messages,
            active_conversations_24h=active_conversations_24h,
            average_messages_per_conversation=round(avg_messages, 2),
            most_recent_conversation=most_recent
        )
    
    def search_conversations(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[ConversationWithMessages]:
        """Search conversations by content or title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Search in conversation titles and message content
            where_conditions = ["(c.title LIKE ? OR m.content LIKE ?)"]
            params = [f"%{query}%", f"%{query}%"]
            
            if user_id:
                where_conditions.append("c.user_id = ?")
                params.append(user_id)
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            search_query = f"""
                SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.user_id, c.metadata, c.message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                {where_clause}
                ORDER BY c.updated_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor = conn.execute(search_query, params)
            conversations = []
            
            for row in cursor.fetchall():
                # Get messages for each conversation
                messages = self._get_conversation_messages(conn, row['id'])
                
                conversations.append(ConversationWithMessages(
                    id=row['id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                    user_id=row['user_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    message_count=row['message_count'] or 0,
                    messages=messages
                ))
        
        return conversations
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Delete conversations older than specified days."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            cursor = conn.execute("""
                DELETE FROM conversations 
                WHERE updated_at < ?
            """, (cutoff_date,))
            
            conn.commit()
            return cursor.rowcount
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get table info
            tables = {}
            for table_name in ['conversations', 'messages']:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = [dict(row) for row in cursor.fetchall()]
                
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                count = cursor.fetchone()['count']
                
                tables[table_name] = {
                    'columns': columns,
                    'row_count': count
                }
            
            # Get database size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            db_size_bytes = page_count * page_size
            
        return {
            'database_path': self.db_path,
            'size_bytes': db_size_bytes,
            'size_mb': round(db_size_bytes / (1024 * 1024), 2),
            'tables': tables,
            'created_at': datetime.utcnow().isoformat()
        }