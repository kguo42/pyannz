sdss example data are downloaded from the origianl code package 
link: https://www.homepages.ucl.ac.uk/~ucapola/annz.html

There are 11 columns in each dataset:
col1-col5 : ugriz magnitude
col6-col10 : ugriz magnitude error
col 11: spectroscopic redshift

len(train) = 5000
len(validation) = 1000
len(test) = 6000

example way to read:

import astropy.table.Table
data = Table.read('./sdss.ugriz.train', format = 'ascii')

input_data_x = data['col1', 'col2','col3','col4','col5'] #ugriz photometry, from ANNz readme
input_data_y = data['col11'] #redshift