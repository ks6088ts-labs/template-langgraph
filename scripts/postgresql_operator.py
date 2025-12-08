import logging

import typer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="PostgreSQL operator CLI",
)

# Set up logging
logger = get_logger(__name__)


def get_database_url(
    user: str,
    password: str,
    host: str,
    port: int,
    dbname: str,
    use_entra_id: bool,
    sslmode: str,
) -> str:
    if use_entra_id:
        logger.info("Using Microsoft Entra ID authentication for PostgreSQL.")
        credential = DefaultAzureCredential()
        password = credential.get_token("https://ossrdbms-aad.database.windows.net/.default").token

    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"


@app.command(
    help="Perform basic PostgreSQL operations using SQLAlchemy ORM ref. https://zenn.dev/yuto_mo/articles/5c7ea612947cf2",
)
def test(
    user: str = typer.Option(
        "user",
        "--user",
        "-u",
        help="Database user",
        envvar="DB_USER",
    ),
    password: str = typer.Option(
        "password",
        "--password",
        "-p",
        help="Database password",
        envvar="DB_PASSWORD",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        "-h",
        help="Database host",
        envvar="DB_HOST",
    ),
    port: int = typer.Option(
        5432,
        "--port",
        "-P",
        help="Database port",
        envvar="DB_PORT",
    ),
    dbname: str = typer.Option(
        "db",
        "--dbname",
        "-n",
        help="Database name",
        envvar="DB_NAME",
    ),
    use_entra_id: bool = typer.Option(
        False,
        "--use-entra-id",
        help="Use Azure Entra ID for authentication",
    ),
    sslmode: str = typer.Option(
        "prefer",
        "--sslmode",
        help="SSL mode for PostgreSQL connection",
        envvar="DB_SSLMODE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Connecting to PostgreSQL database...")

    database_url = get_database_url(
        user=user,
        password=password,
        host=host,
        port=port,
        dbname=dbname,
        use_entra_id=use_entra_id,
        sslmode=sslmode,
    )

    # 1. Create a SQLAlchemy engine
    engine = create_engine(database_url)

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Test query result:", result.fetchone())

    # 3. Declarativeベースを作成（ORM用のクラス定義の基底クラス）
    Base = declarative_base()

    # 4. モデルクラスの定義（usersテーブルに対応）
    class User(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(50))
        fullname = Column(String(50))
        nickname = Column(String(50))

        def __repr__(self):
            return f"<User(name={self.name}, fullname={self.fullname}, nickname={self.nickname})>"

    # 5. テーブルをデータベースに作成
    Base.metadata.create_all(engine)

    # 6. セッションの作成(実際にDBと通信するためのobj)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 7. 新しいユーザの追加（Create）
    new_user = User(name="Alice", fullname="Alice Wonderland", nickname="alice")
    session.add(new_user)
    session.commit()  # 変更をコミットしてデータベースに反映

    # 8. データのクエリ（Read）
    user = session.query(User).filter_by(name="Alice").first()
    print("取得したユーザ：", user)

    # 9. ユーザ情報の更新（Update）
    user.nickname = "ally"
    session.commit()  # 更新をコミット

    # 10. ユーザ情報の削除（Delete）
    session.delete(user)
    session.commit()


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
