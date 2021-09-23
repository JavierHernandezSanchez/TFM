import requests
import os
import os.path

data_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
data_path = r'../data'
file_path = r'../data/uci_data.xls'

r = requests.get(data_url, allow_redirects=True)


if not os.path.isfile(data_path):
    try:
        os.mkdir(data_path)
    except OSError:
        print ("Creation of the directory %s failed" % data_path)
    else:
        print ("Successfully created the directory %s " % data_path)

if os.path.isfile(file_path):
    print ("File exist")
else:
    print ("File not exist")
    open(file_path, 'wb').write(r.content)
