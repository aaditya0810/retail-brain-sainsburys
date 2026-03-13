"""
Retail Brain — API Authentication Module
Handles JWT token generation, password hashing, and role-based access control (RBAC).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from database import get_db_session
from db_models import User, AuditLog
from logger import get_logger

logger = get_logger("api.auth")
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    store_id: Optional[str] = None

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "Viewer"  # Admin, StoreManager, Viewer
    store_id: Optional[str] = "SBY-LON-001"

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    role: str
    store_id: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True


# ── Utilities ─────────────────────────────────────────────────────────────────
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# ── Dependencies ──────────────────────────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    session: Session = Depends(get_db_session)
) -> User:
    """Validate JWT and return the database user object."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email, role=payload.get("role"), store_id=payload.get("store_id"))
    except JWTError:
        raise credentials_exception
        
    user = session.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class RequireRole:
    """Dependency injection class for Role-Based Access Control."""
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Requires one of: {self.allowed_roles}"
            )
        return user


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_db_session)
):
    """OAuth2 compatible token login, get an access token for future requests."""
    user = session.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role, "store_id": user.store_id}, 
        expires_delta=access_token_expires
    )
    
    # Audit log
    audit = AuditLog(action="user_login", entity_type="User", entity_id=str(user.id), user_id=str(user.id))
    session.add(audit)
    session.commit()
    
    logger.info(f"User login successful: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Return the currently authenticated user."""
    return current_user


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate, 
    session: Session = Depends(get_db_session),
    current_user: User = Depends(RequireRole(["Admin"]))
):
    """
    Register a new user. 
    RBAC: Only Admins can register new users to prevent public signups.
    """
    existing_user = session.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        store_id=user_data.store_id
    )
    
    session.add(new_user)
    
    # Audit log
    audit = AuditLog(
        action="user_registered", 
        entity_type="User", 
        details=f"Created {user_data.email} with role {user_data.role}", 
        user_id=str(current_user.id)
    )
    session.add(audit)
    session.commit()
    session.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.email} (Role: {new_user.role})")
    return new_user


def init_default_admin(session: Session):
    """Create a default admin user if no users exist."""
    if session.query(User).count() == 0:
        admin = User(
            email="admin@retailbrain.ai",
            hashed_password=get_password_hash("admin123"), # Change in prod
            full_name="System Admin",
            role="Admin",
            store_id="*"
        )
        session.add(admin)
        session.commit()
        logger.info("Created default Admin user: admin@retailbrain.ai / admin123")
