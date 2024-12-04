from sqlalchemy import text

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


# # test
#     with db.engine.connect() as con:
#         re = con.execute(text("select 1"))
#         print(re.fetchone())