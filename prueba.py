import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Configuración ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TFG_CHAT.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── Modelos ────────────────────────────────────────────────────────────────────
class UserProfile(Base):
    __tablename__ = "profile"
    id            = Column(String, primary_key=True)
    name          = Column(String)
    relationships = Column(String)
    goals         = Column(String)
    topics        = Column(String)
    sessions      = relationship("ChatSession", back_populates="user")

class ChatSession(Base):
    __tablename__        = "sessions"
    id                   = Column(String, primary_key=True)
    name                 = Column(String)
    created_at           = Column(DateTime)
    conversation_summary = Column(Text)
    user_id              = Column(String, ForeignKey("profile.id"))
    messages             = relationship("ChatMessage", cascade="all, delete-orphan", back_populates="session")
    events               = relationship("ImportantEvents", cascade="all, delete-orphan", back_populates="session")
    user                 = relationship("UserProfile", back_populates="sessions", foreign_keys=[user_id])

class ChatMessage(Base):
    __tablename__    = "messages"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(String, ForeignKey("sessions.id"))
    role             = Column(String)
    content          = Column(String)
    detected_emotion = Column(String)
    date             = Column(DateTime)
    session          = relationship("ChatSession", back_populates="messages")

class ImportantEvents(Base):
    __tablename__ = "events"
    id            = Column(String, primary_key=True)
    session_id    = Column(String, ForeignKey("sessions.id"))
    date          = Column(DateTime)
    event         = Column(String)
    type          = Column(String)
    importance    = Column(String)
    session       = relationship("ChatSession", back_populates="events")

# ── Helpers ────────────────────────────────────────────────────────────────────
W = 70
def sep(char="="):  print(char * W)
def line(char="-"): print(char * W)
def val(v, empty="(vacío)"): return v if v else empty
def trunc(s, n=60):
    s = str(s).replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s

# ── Inspector ──────────────────────────────────────────────────────────────────
def inspect():
    if not os.path.exists(DB_PATH):
        print(f"\n❌  No se encontró la base de datos en:\n   {DB_PATH}\n")
        return

    db = SessionLocal()
    try:
        sep()
        print(f"{'🔍  INSPECTOR DE BASE DE DATOS':^{W}}")
        print(f"{'TFG_CHAT':^{W}}")
        sep()
        print(f"📁  Ruta : {DB_PATH}")
        print(f"📅  Hora : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sep()

        # ── PERFILES ──────────────────────────────────────────────────────────
        profiles = db.query(UserProfile).all()
        print(f"\n👤  PERFILES DE USUARIO  ({len(profiles)} registros)")
        line()
        if not profiles:
            print("   (Tabla vacía)")
        for p in profiles:
            print(f"  ID            : {p.id}")
            print(f"  Nombre        : {val(p.name)}")
            print(f"  Relaciones    : {val(p.relationships)}")
            print(f"  Metas         : {val(p.goals)}")
            print(f"  Temas         : {val(p.topics)}")
            n_ses = len(p.sessions)
            print(f"  Sesiones      : {n_ses}")
            line("·")

        # ── SESIONES ──────────────────────────────────────────────────────────
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        print(f"\n💬  SESIONES  ({len(sessions)} registros)")
        line()
        if not sessions:
            print("   (Tabla vacía)")
        for s in sessions:
            print(f"  ID            : {s.id}")
            print(f"  Nombre        : {val(s.name)}")
            print(f"  Creada        : {val(s.created_at)}")
            print(f"  Usuario ID    : {val(s.user_id)}")
            resumen = trunc(s.conversation_summary) if s.conversation_summary else "(sin resumen)"
            print(f"  Resumen       : {resumen}")
            print(f"  Mensajes      : {len(s.messages)}")
            print(f"  Eventos       : {len(s.events)}")
            line("·")

        # ── MENSAJES ──────────────────────────────────────────────────────────
        messages = db.query(ChatMessage).order_by(ChatMessage.date.asc()).all()
        print(f"\n📨  MENSAJES  ({len(messages)} registros)")
        line()
        if not messages:
            print("   (Tabla vacía)")
        for m in messages:
            role_icon = "🧑" if m.role == "user" else "🤖"
            print(f"  {role_icon} [{m.date}]  sesión: {m.session_id[:8]}…")
            print(f"     Contenido : {trunc(m.content)}")
            if m.detected_emotion:
                print(f"     Emoción   : {trunc(m.detected_emotion, 80)}")
            line("·")

        # ── EVENTOS ───────────────────────────────────────────────────────────
        events = db.query(ImportantEvents).order_by(ImportantEvents.date.asc()).all()
        print(f"\n📌  EVENTOS IMPORTANTES  ({len(events)} registros)")
        line()
        if not events:
            print("   (Tabla vacía)")
        for e in events:
            print(f"  ID            : {e.id}")
            print(f"  Sesión        : {e.session_id[:8]}…")
            print(f"  Fecha         : {val(e.date)}")
            print(f"  Evento        : {trunc(e.event)}")
            print(f"  Tipo          : {val(e.type)}")
            print(f"  Importancia   : {val(e.importance)}")
            line("·")

        # ── RESUMEN FINAL ─────────────────────────────────────────────────────
        sep()
        print(f"\n📊  RESUMEN GLOBAL")
        line()
        print(f"  Perfiles   : {len(profiles)}")
        print(f"  Sesiones   : {len(sessions)}")
        print(f"  Mensajes   : {len(messages)}")
        print(f"  Eventos    : {len(events)}")

        perfiles_vacios = sum(
            1 for p in profiles
            if not p.relationships and not p.goals and not p.topics
        )
        if perfiles_vacios:
            print(f"\n  ⚠️  {perfiles_vacios} perfil(es) sin relaciones/metas/temas guardados.")
        else:
            print(f"\n  ✅  Todos los perfiles tienen datos biográficos.")

        sep()

    finally:
        db.close()

if __name__ == "__main__":
    inspect()