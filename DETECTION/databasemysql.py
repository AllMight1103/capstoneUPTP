import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

class ALPRDatabase:
    def __init__(self):
        self.host = "alprdb.cj66c4g68m7i.us-east-1.rds.amazonaws.com"
        self.username = "fawaen"
        self.password = "Max200474"
        self.database = "alprdb"
        self.init_db()

    def connect(self):
        """Establish a connection to the MySQL database."""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database
            )
            return conn
        except mysql.connector.Error as err:
            print("Error:", err)
            raise

    def init_db(self):
        conn = self.connect()
        try:
            c = conn.cursor()
            c.execute('''DROP TABLE IF EXISTS ALPR''')
            c.execute('''CREATE TABLE IF NOT EXISTS ALPR
                         (id INT AUTO_INCREMENT PRIMARY KEY, plate_text VARCHAR(10), detection_time DATETIME)''')
            conn.commit()
        finally:
            c.close()
            conn.close()

    def insert_license_plate(self, plate_text, detection_time):
        """Insert or update a license plate record."""
        conn = self.connect()
        try:
            c = conn.cursor()
            c.execute("SELECT id FROM ALPR WHERE plate_text = %s", (plate_text,))
            result = c.fetchone()
            if result:
                c.execute("UPDATE ALPR SET detection_time = %s WHERE id = %s", (detection_time, result[0]))
            else:
                c.execute("INSERT INTO ALPR (plate_text, detection_time) VALUES (%s, %s)", (plate_text, detection_time))
            conn.commit()
        finally:
            c.close()
            conn.close()

    def select_license_plates(self):
        """Select all license plate records."""
        conn = self.connect()
        try:
            c = conn.cursor(dictionary=True)
            c.execute("SELECT * FROM ALPR")
            results = c.fetchall()
            return results
        finally:
            c.close()
            conn.close()

        
# Usage example:
# Replace 'your_host', 'your_username', 'your_password', and 'your_database' with your actual RDS credentials
db = ALPRDatabase()

# plate_text = 'ABC124'
# detection_time = datetime.now()
# db.insert_license_plate(plate_text, detection_time)

# Selecting rows
rows = db.select_license_plates()
for row in rows:
    print(row)