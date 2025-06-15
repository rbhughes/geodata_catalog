import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType


def string_to_iso_date(df: DataFrame, string_col: str, new_col: str) -> DataFrame:
    def format_timestamp(col_expr):
        return F.to_timestamp(
            F.regexp_replace(col_expr, r"\.\d+", ""), "yyyy-MM-dd'T'HH:mm:ss"
        )

    # Get the actual data type object
    schema_field = next(field for field in df.schema.fields if field.name == string_col)

    if isinstance(schema_field.dataType, ArrayType):
        # For array columns, apply transformation to each element
        transformation = F.transform(F.col(string_col), format_timestamp)
    else:
        # For single value columns, apply transformation directly
        transformation = format_timestamp(F.col(string_col))

    return df.withColumn(new_col, transformation)


# def generate_hash(df: DataFrame, hash_col_name: str, kind: str, *values) -> DataFrame:
#     col_expressions = [F.lit(str(kind))]

#     for value in values:
#         try:
#             col_ref = F.col(str(value))
#             # Get the actual data type object for proper type checking
#             schema_field = next(
#                 field for field in df.schema.fields if field.name == str(value)
#             )

#             if isinstance(schema_field.dataType, ArrayType):
#                 # For array columns, convert array to string representation
#                 col_expressions.append(
#                     F.concat(F.lit("["), F.array_join(col_ref, ","), F.lit("]"))
#                 )
#             else:
#                 # For single values, cast to string
#                 col_expressions.append(F.col(str(value)).cast("string"))
#         except (StopIteration, AttributeError):
#             # Column doesn't exist in schema, treat as literal value
#             col_expressions.append(F.lit(str(value)))

#     return df.withColumn(
#         hash_col_name, F.sha2(F.concat_ws("||", *col_expressions), 256)
#     )


def generate_hash(df: DataFrame, hash_col_name: str, kind: str, *values) -> DataFrame:
    # Define the transformation logic once
    def process_value(value, df_schema):
        try:
            col_ref = F.col(str(value))
            # Get the actual data type object for proper type checking
            schema_field = next(
                field for field in df_schema.fields if field.name == str(value)
            )

            if isinstance(schema_field.dataType, ArrayType):
                # For array columns, convert array to string representation
                return F.concat(F.lit("["), F.array_join(col_ref, ","), F.lit("]"))
            else:
                # For single values, cast to string
                return F.col(str(value)).cast("string")
        except (StopIteration, AttributeError):
            # Column doesn't exist in schema, treat as literal value
            return F.lit(str(value))

    # Build column expressions using the extracted logic
    col_expressions = [F.lit(str(kind))]
    col_expressions.extend(process_value(value, df.schema) for value in values)

    return df.withColumn(
        hash_col_name, F.sha2(F.concat_ws("||", *col_expressions), 256)
    )


def replace_10e30_with_null(df: DataFrame, col_name: str, new_col: str) -> DataFrame:
    # Petra tends to use 1e+30 for doubles.
    def replace_value(col_expr):
        return F.when(col_expr == F.lit(1e30), F.lit(None)).otherwise(col_expr)

    # Get the actual data type object
    schema_field = next(field for field in df.schema.fields if field.name == col_name)

    if isinstance(schema_field.dataType, ArrayType):
        # For array columns, apply transformation to each element
        transformation = F.transform(F.col(col_name), replace_value)
    else:
        # For single value columns, apply transformation directly
        transformation = replace_value(F.col(col_name))

    return df.withColumn(new_col, transformation)
