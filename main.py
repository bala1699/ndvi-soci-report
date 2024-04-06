from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import rasterio
import pandas as pd
import csv
import json
from fastapi import FastAPI, Response, UploadFile


app = FastAPI()


input_blue  = 'LC08_B2.TIF'
input_green = 'LC08_B3.TIF'
input_red   = 'LC08_B4.TIF'
input_nir   = 'LC08_B5.TIF'

extent_blue  = 'clipped_blue.tif'
extent_green = 'clipped_green.tif'
extent_red   = 'clipped_red.tif'
extent_nir   = 'clipped_nir.tif'


utm11_srs = osr.SpatialReference()
utm11_srs.SetWellKnownGeogCS("WGS84")
utm11_srs.SetUTM(44, True)
bounds_utm = (232639.673, 1291318.668, 236575.025, 1288916.921)

gdal.Warp(extent_blue, input_blue, outputBounds=bounds_utm, outputBoundsSRS=utm11_srs)
gdal.Warp(extent_green, input_green, outputBounds=bounds_utm, outputBoundsSRS=utm11_srs)
gdal.Warp(extent_red , input_red, outputBounds=bounds_utm, outputBoundsSRS=utm11_srs)
gdal.Warp(extent_nir, input_nir, outputBounds=bounds_utm, outputBoundsSRS=utm11_srs)


def tif_to_csv(input_tif, output_csv):
    with rasterio.open(input_tif) as src:
        
        data = src.read(1) 
        width = src.width
        height = src.height
        transform = src.transform

    
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
        
            writer.writerow(['x', 'y', 'value'])
        
            for y in range(height):
                for x in range(width):
                
                    lon, lat = src.xy(y, x)
                
                    writer.writerow([lon, lat, data[y, x]])


input_tif_blue  = 'clipped_blue.tif'
input_tif_green = 'clipped_green.tif'
input_tif_red   = 'clipped_red.tif'
input_tif_nir   = 'clipped_nir.tif'

output_csv_blue  = 'csv_blue.csv'
output_csv_green = 'csv_green.csv'
output_csv_red   = 'csv_red.csv'
output_csv_nir  = 'csv_nir.csv'


tif_to_csv(input_tif_blue, output_csv_blue)
tif_to_csv(input_tif_green, output_csv_green)
tif_to_csv(input_tif_red, output_csv_red)
tif_to_csv(input_tif_nir, output_csv_nir)


def csv_to_json(input_csv, output_json):
    data = {}
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        key=1
        for row in reader:
            
            data[key]=row
            key = key+1

    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)


input_csv_blue  = 'csv_blue.csv'
input_csv_green = 'csv_green.csv'
input_csv_red   = 'csv_red.csv'
input_csv_nir   = 'csv_nir.csv'

output_json_blue  = 'json_blue.json'
output_json_green = 'json_green.json'
output_json_red   = 'json_red.json'
output_json_nir   = 'json_nir.json'



csv_to_json(input_csv_blue, output_json_blue)
csv_to_json(input_csv_green, output_json_green)
csv_to_json(input_csv_red, output_json_red)
csv_to_json(input_csv_nir, output_json_nir)



with open(output_json_blue, 'r') as f1:
        data_blue = json.load(f1)

    
with open(output_json_green, 'r') as f2:
        data_green = json.load(f2)

with open(output_json_red, 'r') as f1:
        data_red = json.load(f1)

    
with open(output_json_nir, 'r') as f2:
        data_nir = json.load(f2)


ndvi = data_nir

#Below is the code to make sure the coordinates match. But blocked it because my laptop couldn't handle it.
for key1 in data_nir:
    for key2 in data_red:

        if data_nir[key1]["x"] == data_red[key2]["x"]:
            if data_nir[key1]["y"] == data_red[key2]["y"]:

                num   = float(data_nir[key1]["value"]) - float(data_red[key2]["value"])
                denom = float(data_nir[key1]["value"]) + float(data_red[key2]["value"])

                
                ndvi[key1]["ndvi"]= num/denom
                del ndvi[key1]["value"]
                
        else:
            continue      
        
            
                
'''
for key in ndvi:
    num   = float(data_nir[key]["value"]) - float(data_red[key]["value"])
    denom = float(data_nir[key]["value"]) + float(data_red[key]["value"])

                
    ndvi[key]["ndvi"]= num/denom
    del ndvi[key]["value"]
    
'''
                
ndvi_values = [ndvi[key]["ndvi"] for key in ndvi]

#The above code is for calculating ndvi with crosschecking the coordinates. It also can be done without checking as given below which is faster.

soci = data_blue


for key1 in data_blue:
    for key2 in data_red:
        if  data_blue[key1]["x"] == data_red[key2]["x"] and data_blue[key1]["y"] == data_red[key2]["y"]:

            for key3 in data_green:
                if  data_blue[key1]["x"] == data_green[key3]["x"]  and data_blue[key1]["y"] == data_green[key3]["y"]:
                    
                    num   = float(data_blue[key1]["value"])
                    denom = float(data_red[key2]["value"]) * float(data_green[key3]["value"])

                    soci[key1]["soci"]= num/denom
                    del soci[key1]["value"]
                        
                else:
                    continue    
        else:
            continue   

               
#The above code is for calculating soci with crosschecking the coordinates. It also can be done without checking as given below which is faster.

'''
for key in soci:
    num   = float(data_blue[key]["value"])
    denom = float(data_red[key]["value"]) * float(data_green[key]["value"])

    soci[key]["soci"]= num/denom
    del soci[key]["value"]
'''
     


soci_values = [soci[key]["soci"] for key in soci]


                
                        
def chart_ndvi(input_data):
     
    plt.hist(input_data, bins=10, edgecolor='black')
    plt.xlabel('NDVI Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of NDVI')
    plt.grid(True)

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.clf()
    return img_data.getvalue()

def chart_soci(input_data):
     
    plt.hist(input_data, bins=10, edgecolor='black')
    plt.xlabel('SOCI Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of SOCI')
    plt.grid(True)

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.clf()
    return img_data.getvalue()

def map_ndvi(input_data):

    new_shape=(80,131)
    input_data=np.reshape(input_data, new_shape)

    plt.imshow(input_data, cmap = 'YlGn', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap')
    

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.clf()
    return img_data.getvalue()

def map_soci(input_data):

    new_shape=(80,131)
    input_data=np.reshape(input_data, new_shape)

    plt.imshow(input_data, cmap = 'PuBu', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap')
    

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.clf()
    return img_data.getvalue()


app.listen(process.env.PORT || 3000, function(){
  console.log("Express server listening on port %d in %s mode", this.address().port, app.settings.env);
});

@app.get("/get_ndvi_values/")
def get_ndvi_values():

    return ndvi
    

@app.get("/get_ndvi_chart/")
def get_ndvi_chart():

    plot_chart_ndvi = chart_ndvi(ndvi_values)

    return Response(content=plot_chart_ndvi, media_type="image/png")


@app.get("/get_ndvi_heatmap/")
def get_ndvi_heatmap():

    plot_map_ndvi = map_ndvi(ndvi_values)

    return Response(content=plot_map_ndvi, media_type="image/png")

@app.get("/get_soci_values/")
def get_soci_values():

    return soci
    

@app.get("/get_soci_chart/")
def get_soci_chart():

    plot_chart_soci = chart_soci(soci_values)

    return Response(content=plot_chart_soci, media_type="image/png")


@app.get("/get_soci_heatmap/")
def get_soci_heatmap():

    plot_map_soci = map_soci(soci_values)

    return Response(content=plot_map_soci, media_type="image/png")

@app.get("/getpixelcount/")
def get_pixel_count():
    count = len(soci_values)
    return {"Total number of pixels" : count}
