# running only ML algorithms on the whole dataset
python conventional.py

# decomposing dataset into OLAP and OLTP
python make_oltp_data1.py
python make_oltp_data2.py
python make_oltp_data3.py
python make_oltp_data4.py
python make_oltp_data5.py

# run
python olap.py  # set serial=1 in in line #13
python oltp.py  # set serial=1 in in line #13

python olap.py  # set serial=2 in in line #13
python oltp.py  # set serial=2 in in line #13

python olap.py  # set serial=3 in in line #13
python oltp.py  # set serial=3 in in line #13

python olap.py  # set serial=4 in in line #13
python oltp.py  # set serial=4 in in line #13

python olap.py  # set serial=5 in in line #13
python oltp.py  # set serial=5 in in line #13

# result metrics
python result.py # set serial=1 in in line #13
python result.py # set serial=2 in in line #13
python result.py # set serial=3 in in line #13
python result.py # set serial=4 in in line #13
python result.py # set serial=5 in in line #13