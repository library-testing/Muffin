from pathlib import Path
import shutil
import sqlite3
import sys

dataset_name = sys.argv[1] # e.g. 'fashion_mnist'

output_path = Path(sys.argv[0]).resolve().parent / f'{dataset_name}_output'
if output_path.exists():
    shutil.rmtree(output_path)


report_path = Path(sys.argv[0]).resolve().parent / f'{dataset_name}_report'
if report_path.exists():
    shutil.rmtree(report_path)


db_path = Path(sys.argv[0]).resolve().parent / f'{dataset_name}.db'
if db_path.exists():
    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    CLEAR_MODEL = '''drop table model'''
    CLEAR_INCONS = '''drop table inconsistency'''
    CLEAR_MAP = '''drop table localization_map'''
    CREATE_MODEL = '''CREATE TABLE "model" (
                        "dataset_name"	TEXT,
                        "node_num"	INTEGER NOT NULL,
                        "generate_fail_backends"	TEXT,
                        "crash_backends"	TEXT,
                        "nan_backends"	TEXT,
                        "inf_backends"	TEXT,
                        "loss_func"	TEXT,
                        "optimizer"	TEXT,
                        "status"	TEXT
                    )
                    '''
    CREATE_INCONS = '''CREATE TABLE "inconsistency" (
                        "model_id"	INTEGER NOT NULL,
                        "backend_pair"	TEXT NOT NULL,
                        "model_output_delta"	REAL,
                        "loss_delta"	REAL,
                        "loss_grads_delta"	REAL,
                        "weights_delta"	REAL,
                        PRIMARY KEY("model_id","backend_pair")
                    )'''
    CREATE_MAP = '''CREATE TABLE "localization_map" (
                    "incons_id"	INTEGER NOT NULL,
                    "layer_name"	TEXT NOT NULL,
                    "outputs_delta"	REAL,
                    "outputs_R"	REAL,
                    "gradients_delta"	REAL,
                    "gradients_R"	REAL,
                    "weights_delta"	REAL,
                    "inbound_layers"	TEXT,
                    PRIMARY KEY("incons_id","layer_name")
                )'''

    conn.execute(CLEAR_MODEL)
    conn.execute(CLEAR_INCONS)
    conn.execute(CLEAR_MAP)
    conn.execute(CREATE_MODEL)
    conn.execute(CREATE_INCONS)
    conn.execute(CREATE_MAP)
    conn.commit()

print("Done!")
