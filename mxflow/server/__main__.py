from mxflow.server.db import db
from mxflow.server.auth import get_password_hash


def create_admin():
    user = db.get_user("admin")
    if user is None:
        user = db.create_user(
            username="admin",
            password=get_password_hash("admin"),
            role="admin",
        )
        print(
            "User 'admin' created:",
            "id:",
            user.id,
            ", Username:",
            user.username,
        )
    else:
        print(
            "User 'admin' existing:",
            "id:",
            user.id,
            ", Username:",
            user.username,
        )


def create_guest():
    user = db.get_user("guest")
    if user is None:
        user = db.create_user(
            username="guest",
            password=get_password_hash("guest"),
        )
        print(
            "User 'admin' created:",
            "id:",
            user.id,
            ", Username:",
            user.username,
        )
    else:
        print(
            "User 'admin' existing:",
            "id:",
            user.id,
            ", Username:",
            user.username,
        )


if __name__ == "__main__":
    create_admin()
    create_guest()
