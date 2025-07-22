
#BasyTec has the following header 
#Time[s],DataSet,t-Step[s],t-Set[s],Line,U[V],I[A],Ah[Ah],Ah-Step,Wh[Wh],Cyc-Count,State
# we will map this header to a uniform ontology based naming convention : 
# original_column_name: standardized_ontology_column_name


BASYTEC_COLUMN_MAPPING = {
    "Time[s]": "time_s",
    "DataSet": "dataset_idx",
    "t-Step[s]": "step_time_s",
    "t-Set[s]": "set_time_s",
    "Line": "line_idx",
    "U[V]": "voltage_v",
    "I[A]": "current_a",
    "Ah[Ah]": "charge_ah",
    "Ah-Step": "step_charge_ah",
    "Wh[Wh]": "energy_wh",
    "Cyc-Count": "cycle_index",
    "State": "state_code",
}

BASYTEC_METADATA_MAPPING = {
    "battery": "cell_chemistry",
    "testplan": "test_plan",
    "testchannel": "test_channel",
    "operator (test)": "operator_test",
    "operator (data converting)": "operator_data_converting",
    "start_of_test": "start_of_test",
    "end_of_test": "end_of_test",
    "date_and_time_of_data_converting": "data_extraction_time",
    # add more as needed!
}
