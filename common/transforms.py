# import pyspark.sql.functions as F
# from pyspark.sql import DataFrame

# def string_to_iso_date(df: DataFrame, string_col: str, new_col: str) -> DataFrame:
#     return df.withColumn(
#         new_col,
#         F.to_timestamp(F.col(string_col), "yyyy-MM-dd'T'HH:mm:ss")
#     )


# def generate_hash(df: DataFrame, hash_col_name: str, kind: str, *values) -> DataFrame:
#     col_expressions = [F.lit(str(kind))]
#     for value in values:
#         try:
#             col_expressions.append(F.col(str(value)))
#         except:
#             col_expressions.append(F.lit(str(value)))

#     return df.withColumn(
#         hash_col_name,
#         F.sha2(F.concat_ws("||", *col_expressions), 256)
#     )


import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType


def string_to_iso_date(df: DataFrame, string_col: str, new_col: str) -> DataFrame:
    # Get the actual data type object, not just the string representation
    schema_field = next(field for field in df.schema.fields if field.name == string_col)

    if isinstance(schema_field.dataType, ArrayType):
        # For array columns, use transform to apply the conversion to each element
        return df.withColumn(
            new_col,
            F.transform(
                F.col(string_col),
                lambda x: F.to_timestamp(
                    F.regexp_replace(x, r"\.\d+", ""), "yyyy-MM-dd'T'HH:mm:ss"
                ),
            ),
        )
    else:
        # For single value columns, strip microseconds and convert
        return df.withColumn(
            new_col,
            F.to_timestamp(
                F.regexp_replace(F.col(string_col), r"\.\d+", ""),
                "yyyy-MM-dd'T'HH:mm:ss",
            ),
        )


def generate_hash(df: DataFrame, hash_col_name: str, kind: str, *values) -> DataFrame:
    col_expressions = [F.lit(str(kind))]

    for value in values:
        try:
            col_ref = F.col(str(value))
            # Get the actual data type object for proper type checking
            schema_field = next(
                field for field in df.schema.fields if field.name == str(value)
            )

            if isinstance(schema_field.dataType, ArrayType):
                # For array columns, convert array to string representation
                col_expressions.append(
                    F.concat(F.lit("["), F.array_join(col_ref, ","), F.lit("]"))
                )
            else:
                # For single values, cast to string
                col_expressions.append(F.col(str(value)).cast("string"))
        except (StopIteration, AttributeError):
            # Column doesn't exist in schema, treat as literal value
            col_expressions.append(F.lit(str(value)))

    return df.withColumn(
        hash_col_name, F.sha2(F.concat_ws("||", *col_expressions), 256)
    )
