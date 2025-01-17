# from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, LargeBinary, DateTime, JSON
# from sqlalchemy.sql import func
# from base import Base

# class ChatBotMedia(Base):
#     __tablename__ = 'chatbot_media'

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     file_path = Column(String, nullable=False)
#     file_type = Column(String, nullable=False)
#     user_id = Column(String, nullable=False)
#     content_metadata = Column(String, nullable=False)
#     upload_timestamp = Column(DateTime, nullable=True, default=func.now())

    # def __repr__(self):
    #     return f"ChatBotMedia(id={self.id}, file_path={self.file_path}, file_type={self.file_type}, user_id={self.user_id}, content_metadata={self.content_metadata}, upload_timestamp={self.upload_timestamp})"

from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, LargeBinary, DateTime, JSON
from sqlalchemy.sql import func
from base import Base

class ChatBotMedia(Base):
    __tablename__ = 'chatbot_media'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    content_metadata = Column(JSON, nullable=False)  # Store metadata as JSON
    file_content = Column(LargeBinary, nullable=False)  # Store file content as binary
    upload_timestamp = Column(DateTime, nullable=True, default=func.now())