from datetime import datetime
import json
import uuid
from collections import Counter
 
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DATABASE_URL = "sqlite:///TFG_CHAT.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


Base = declarative_base()



class ChatSession(Base):
    __tablename__= "sessions"
    
    id = Column(String, primary_key= True, default = lambda: str (uuid.uuid4()))
    name = Column(String, nullable = True)
    created_at = Column(DateTime, default = datetime.utcnow)
    conversation_summary = Column(Text, nullable = True)
    user_id = Column(String, ForeignKey("profile.id"), nullable=True)
    messages = relationship("ChatMessage", cascade = "all, delete-orphan", back_populates= "session")
    events = relationship("ImportantEvents", cascade = "all, delete-orphan", back_populates= "session")
    user = relationship("UserProfile", back_populates="sessions", foreign_keys=[user_id])
    emotion_summaries = relationship("SessionEmotions", cascade="all, delete-orphan", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key = True, autoincrement = True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable = False)
    role = Column(String, nullable = False)
    content = Column(String, nullable=False)
    detected_emotion = Column(String, nullable = True)
    date = Column(DateTime, default = datetime.utcnow)
    session = relationship("ChatSession", back_populates= "messages")


class ImportantEvents(Base) :
    __tablename__= "events"
    
    id = Column(String, primary_key= True, default = lambda: str (uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable = False)
    date = Column(DateTime, default = datetime.utcnow)
    event = Column(String, nullable = False)
    type = Column(String, nullable = True)
    importance = Column(String, nullable = True)
    session = relationship("ChatSession", back_populates= "events") 
    
class UserProfile(Base):
    __tablename__= "profile"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable = True)
    relationships = Column(String, nullable = True)
    goals = Column(String, nullable = True)
    topics = Column(String, nullable = True)
    hobbies = Column(String, nullable = True)
    sessions = relationship("ChatSession", back_populates="user")

class SessionEmotions(Base):
    __tablename__ = "session_emotions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    dominant_emotion = Column(String, nullable=False)
    emotion_scores = Column(Text, nullable=True)  # JSON con todas las emociones
    date = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="emotion_summaries")



Base.metadata.create_all(engine)   


def create_session(name: str, user_id: str) -> str:
    db = SessionLocal()
    try:
        new_chat = ChatSession(name = name, user_id=user_id)
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        session_id = new_chat.id 
    finally:
        db.close()
    return session_id  



def get_all_sessions() -> list[dict]:
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        return [{"id": s.id, "name": s.name, "user_id": s.user_id} for s in sessions]
    finally:
        db.close()


def delete_session(id:str) -> None:
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == id).first()
        if session:
            db.delete(session)
            db.commit()
    finally:
        db.close()
        

def update_session(session_id:str, new_name= None, new_summary = None) -> None :
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            if new_name is not None:
                session.name = new_name
            if new_summary is not None:
                session.conversation_summary = new_summary
            db.commit()
    finally:
        db.close()
        

def save_message(new_id:str, new_role: str, new_content:str, new_detected_emotion = None) -> None :
    db = SessionLocal()
    try:
        new_message = ChatMessage(session_id = new_id, role = new_role, content = new_content, detected_emotion = new_detected_emotion)
        db.add(new_message)
        db.commit()
        
    finally:
        db.close()



def get_messages(session_id:str) -> list[dict]:
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.date.asc()).all()
        return [
            {
                "id": m.id,
                "session_id": m.session_id,
                "role": m.role,
                "content": m.content,
                "detected_emotion": m.detected_emotion,
                "date": m.date,
            }
            for m in messages
        ]
    finally:
        db.close()
        

def create_event(new_session_id: str, added_event: str, added_type: str = None, added_importance: str = None) -> None:
    db = SessionLocal()
    try:
        new_event = ImportantEvents(session_id = new_session_id, event = added_event, type = added_type, importance = added_importance)
        db.add(new_event)
        db.commit()
        print(f"EVENTO GUARDADO: {added_event}, tipo: {added_type}, importancia: {added_importance}")
        
    finally:
        db.close()
        

def get_events(session_id:str) -> list[dict]:
    db = SessionLocal()
    try:
        events = db.query(ImportantEvents).filter(ImportantEvents.session_id == session_id).order_by(ImportantEvents.date.asc()).all()
        return [
            {
                "id": e.id,
                "session_id": e.session_id,
                "event": e.event,
                "type": e.type,
                "importance": e.importance,
                "date": e.date,
            }
            for e in events
        ]
    finally:
        db.close()
        

def get_conversation_summary(session_id:str) -> str | None:
    db = SessionLocal()
    try:
        conv_sum = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        return conv_sum.conversation_summary if conv_sum else None
    finally:
        db.close()
        

def create_profile(username: str) -> str:
    db = SessionLocal()
    try:
        new_profile = UserProfile(name=username)
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        profile_id = new_profile.id
        return profile_id
    finally:
        db.close()
        
def _merge(existing: str, new: str) -> str:
    existing_items = [x.strip() for x in existing.split("|")] if existing else []
    existing_items = [x for x in existing_items if x]

    new_items = [x.strip() for x in new.replace(";", ",").split(",")]
    new_items = [x for x in new_items if x]

    seen = {item.lower() for item in existing_items}

    merged = list(existing_items)
    for item in new_items:
        key = item.lower()
        if key not in seen:
            merged.append(item)
            seen.add(key)

    return " | ".join(merged)

def update_profile(user_id, name=None, relationships=None, goals=None, topics=None, hobbies=None):
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if user:

            if name: 
                user.name = name


            if relationships:
                user.relationships = _merge(user.relationships, relationships)
            if goals:
                user.goals = _merge(user.goals, goals)
            if topics:
                user.topics = _merge(user.topics, topics)
            if hobbies:
                user.hobbies = _merge(user.hobbies, hobbies)
                
            db.commit()
    finally:
        db.close()
        

def get_or_create_profile(username = None) -> str:
    db = SessionLocal()
    try:
        user = db.query(UserProfile).first()
        if user:
            return user.id

        new_profile = UserProfile(name=username)
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        profile_id = new_profile.id 
        return profile_id
    finally:
        db.close()
        
def get_user_profile_text(user_id: str) -> str:
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not profile:
            return "No hay perfil biográfico disponible."
            
        return (f"Nombre: {profile.name or 'No indicado'}. " 
                f"Relaciones: {profile.relationships or 'No indicadas'}. "
                f"Objetivos: {profile.goals or 'No indicados'}. "
                f"Temas recurrentes: {profile.topics or 'Ninguno'}."
                f"Aficiones: {profile.hobbies or 'No indicadas'}.")
    finally:
        db.close()
        
def delete_profile(user_id: str) -> None:
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
    finally:
        db.close()


def log_session_emotions(session_id: str, emotion_data: list) -> None:
    if not emotion_data:
        return
    dominant = max(emotion_data, key=lambda x: x["score"])["label"]
    db = SessionLocal()
    try:
        entry = db.query(SessionEmotions).filter(
            SessionEmotions.session_id == session_id
        ).first()
        if entry:
            entry.dominant_emotion = dominant
            entry.emotion_scores = json.dumps(emotion_data)
            entry.date = datetime.utcnow()
        else:
            entry = SessionEmotions(
                session_id=session_id,
                dominant_emotion=dominant,
                emotion_scores=json.dumps(emotion_data)
            )
            db.add(entry)
        db.commit()
    finally:
        db.close()


def get_emotion_trends(session_id: str, n: int = 5) -> list[dict]:
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            return []
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == session.user_id, ChatSession.id != session_id)
            .order_by(ChatSession.created_at.desc())
            .limit(n)
            .all()
        )
        resultados = []
        for s in sessions:
            last_entry = (
                db.query(SessionEmotions)
                .filter(SessionEmotions.session_id == s.id)
                .order_by(SessionEmotions.date.desc())
                .first()
            )
            if last_entry:
                resultados.append({
                    "session_id": s.id,
                    "date": last_entry.date,
                    "dominant_emotion": last_entry.dominant_emotion,
                    "emotion_scores": json.loads(last_entry.emotion_scores) if last_entry.emotion_scores else []
                })
        return resultados
    finally:
        db.close()


def predict_next_emotion(session_id: str, n: int = 5) -> str | None:
    from collections import defaultdict

    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            return None

        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == session.user_id)
            .order_by(ChatSession.created_at.desc())
            .limit(n)
            .all()
        )

        acumulado = defaultdict(list)
        for s in sessions:
            last_entry = (
                db.query(SessionEmotions)
                .filter(SessionEmotions.session_id == s.id)
                .order_by(SessionEmotions.date.desc())
                .first()
            )
            if last_entry and last_entry.emotion_scores:
                for emo in json.loads(last_entry.emotion_scores):
                    acumulado[emo["label"]].append(emo["score"])

        if not acumulado:
            return None

        medias = {label: sum(scores) / len(scores) for label, scores in acumulado.items()}
        return max(medias, key=lambda x: medias[x])
    finally:
        db.close()


def delete_event_by_id(event_id: str) -> None:
    db = SessionLocal()
    try:
        event = db.query(ImportantEvents).filter(ImportantEvents.id == event_id).first()
        if event:
            db.delete(event)
            db.commit()
    finally:
        db.close()

def reset_profile_fields(user_id: str, fields: str) -> None:
    CAMPO_MAP = {
        "todo": ["name", "relationships", "goals", "topics", "hobbies"],
        "nombre": ["name"],
        "relaciones": ["relationships"],
        "objetivos": ["goals"],
        "temas": ["topics"],
        "aficiones": ["hobbies"]
    }
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if user:
            for campo in CAMPO_MAP.get(fields.lower(), []):
                setattr(user, campo, None)
            db.commit()
    finally:
        db.close()