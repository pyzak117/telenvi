from osgeo import gdalconst

npdtype_gdalconst={
    'int8'     : gdalconst.GDT_Byte,
    'int16'    : gdalconst.GDT_Int16,
    'int32'    : gdalconst.GDT_Int32,
    'int64'    : gdalconst.GDT_Int64,
    'uint8'    : gdalconst.GDT_Byte,
    'uint16'   : gdalconst.GDT_UInt16,
    'uint32'   : gdalconst.GDT_UInt32,
    'uint64'   : gdalconst.GDT_UInt64,
    'float16'  : gdalconst.GDT_Float32,
    'float32'  : gdalconst.GDT_Float32,
    'float64'  : gdalconst.GDT_Float64,
}

extensions_drivers={
    'tif':'GTiff'
}