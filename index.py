from flask import Flask , render_template , Response
from flask import request

from pic import *
from smoke import *
from pymydb import *
app = Flask(__name__)

def last():
    picture_path = cam2() # take a picture 
    predict = model_predict( picture_path ) # return 1 or 0 | smoke or not
    insert_db( picture_path , predict )
    select_result = select_db( '*' , '2022_BBIBBO')
    print( select_result )

def main():
    app.debug = True
    app.run(host="10.150.150.2", port="8080")


if __name__ == '__main__':
    main()