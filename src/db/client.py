SQL_connection = psycopg2.connect(database = "multisector_pyFi", 
                                  user = "elphaim",
                                  host= 'localhost',
                                  password = "A5QqxYP9js3mLrFDIEUv",
                                  port = 5432)
SQL_cursor = SQL_connection.cursor()
