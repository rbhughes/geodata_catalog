import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def string_to_iso_date(df: DataFrame, string_col: str, new_col: str) -> DataFrame:
    return df.withColumn(
        new_col,
        F.to_timestamp(F.col(string_col), "yyyy-MM-dd'T'HH:mm:ss")
    )


def generate_hash(df: DataFrame, hash_col_name: str, kind: str, *values) -> DataFrame:
    col_expressions = [F.lit(str(kind))]
    for value in values:
        try:
            col_expressions.append(F.col(str(value)))
        except:
            col_expressions.append(F.lit(str(value)))
    
    return df.withColumn(
        hash_col_name,
        F.sha2(F.concat_ws("||", *col_expressions), 256)
    )