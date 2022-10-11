#
from flask import Flask , render_template , Response
from flask import request
from flask_cors import CORS # pip install flask-corsb

from pic import *
from smoke import *
from pymydb import *
app = Flask(__name__)
CORS(
        app,
        resources={r"/api/*": {"origins":"http://localhost:3000"}},
        supports_credentials=True
    )

def last():
    picture_path = cam2() # take a picture 
    predict = model_predict( picture_path ) # return 1 or 0 | smoke or not
    insert_db( picture_path , predict )
    select_result = select_db( '*' , '2022_BBIBBO')
    print( select_result )

@app.route('/api/test')
def test():
    return "test"

@app.route('/api/select_db', methods=['POST'])
def select_DB():
    select_result = select_db('*','2022_BBIBBO')
    return select_result

def main():
    app.debug = True
    app.run(host="10.150.150.2", port="8080")
if __name__ == '__main__':
    main()

# front example
# axios.create({
# 	baseURL : https://test-b.com,
#     withCredentials: true
# })
# try {
#     const res = await axios.post(
#         "http://10.150.150.2:8080/api/test"
#     );
#     console.log(res.data);
# }