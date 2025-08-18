import time
import gspread
import jax
from google.oauth2.service_account import Credentials

from beh.core.registry import *

sheet_id = "1b7f7NInSCiL8j7LPke36DgO3PDBNEITAP3ytK95m_RM"
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("./keys/gsheets_credentials.json", scopes=scopes)
gc = gspread.authorize(creds)

def gsheet_log_row(
    worksheet_name,
    dimension, 
    data_name,
    accelerator,
    chart_key,
    model_type,
    pattern,
    size,
    architecture,
    total_p,
    active_p,
    epochs,
    batch_size,
    time_absolut,
    timing_setup,
    fp,
    fn,):
    '''
    Log a row of results in google sheets using google sheets api.
    '''
    row = [
        dimension,
        data_name,
        accelerator, # GPU / CPU
        chart_key,
        model_type,
        pattern,
        size,
        architecture,
        total_p,
        active_p,
        epochs,
        batch_size,
        time_absolut,
        timing_setup,
        fp,
        fn,
        "FALSE",  # Set approve flag initially to false to avoid push to overleaf
        time.strftime("%Y-%m-%d %H:%M:%S") # Timestamp
    ]
    ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    ws.append_row(row, value_input_option="RAW")

def gsheet_log_results(model_key : str, dimension : int, reg : CoreRegistry, configs, data_name : str):
    '''
    Log all results in google sheets using google sheets api.
    '''
    # Set basic row arguments
    worksheet_name = 'table2'
    accelerator = jax.devices()[0].device_kind
    model_type = configs[model_key]['type']
    
    # Extract type specific information 
    if model_type == 'moe':
        nex = configs[model_key]['nex']
        gate_hid_lay = configs[model_key]['gate_hidden_layer']
        expert_hid_lay = configs[model_key]['expert_hidden_layer']

        m_key = f'{model_key}_dense'
        gate_arch = [dimension] + gate_hid_lay + [nex]
        expert_arch = [dimension] + expert_hid_lay + [1]
        arch = f'G {str(gate_arch).replace(" ", "")}    {nex}x E {str(expert_arch).replace(" ", "")}'
    
    elif model_type == 'mlp':
        m_key = model_key
        hid_lay = configs[m_key]['hidden_layer']
        arch = str([dimension] + hid_lay + [1]).replace(" ", "")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    gsheet_log_row(
        worksheet_name,
        dimension,
        data_name,
        accelerator,
        m_key.split("_")[0]  + "D",
        model_type,
        'Dense',
        "".join(c for c in m_key if c.isdigit()),
        arch,
        float(reg.get(m_key + core_keys['total_parameters_key'])),
        float(reg.get(m_key + core_keys['active_parameters_key'])),
        configs['general']['epochs'],
        int(reg.get(m_key + core_keys['optimal_batch_size_key'])),
        float(reg.get(m_key + core_keys['inf_speed_key'])),
        f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
        float(reg.get(m_key + core_keys['fp_key'])),
        float(reg.get(m_key + core_keys['fn_key']))
    )

    ## Ensure we append all sparse MoE results after all other to 
    ## have them as one block of results together
    if model_type == 'moe':
        m_key = f'{model_key}_sparse'
        gsheet_log_row(
            worksheet_name,
            dimension,
            data_name,
            accelerator,
            m_key.split("_")[0] + "S",
            model_type,
            'Sparse',
            "".join(c for c in m_key if c.isdigit()),
            arch,
            float(reg.get(m_key + core_keys['total_parameters_key'])),
            float(reg.get(m_key + core_keys['active_parameters_key'])),
            configs['general']['epochs'],
            int(reg.get(m_key + core_keys['optimal_batch_size_key'])),
            float(reg.get(m_key + core_keys['inf_speed_key'])),
            f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
            float(reg.get(m_key + core_keys['fp_key'])),
            float(reg.get(m_key + core_keys['fn_key']))
        )