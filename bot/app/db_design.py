from sqlalchemy import Column, Integer, create_engine, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@db:5432/user_context", echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class UserContext(Base):
    __tablename__ = "user_context"
    user_id = Column(Integer, primary_key=True)
    context = Column(MutableDict.as_mutable(JSONB))
    last_usage = Column(TIMESTAMP, server_default=func.now(),
                        onupdate=func.current_timestamp())


Base.metadata.create_all(engine)
