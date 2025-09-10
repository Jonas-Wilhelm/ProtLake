##### This file serves as a template file.
##### Any scorefxn python file starting with "_" will be ignored

# import ...

scorefxn_name = "_template"
description = "optional description"

def score(aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args):
    ...
    # Required
    # should return a dict where the keys are the column names
    # and the values are scalars (one of int, float, bool, str)
    # Available objects:
    #   aa_design: biotite.structure.AtomArray of the design model
    #   aa_af3: biotite.structure.AtomArray of the af3 prediction
    #   meta: dict representation of the deltalake row that is currently processed (also contains all ..._summary_confidences.json data)
    #   confidences: dict that contains all information of the ..._confidences.json that af3 outputs
    #   sc_close_to_het_mask: np.array bool mask to select all sidechains that are close to any ligand (hetero atoms)
    #   CLI_args: argparse.Namespace object with all parsed command line arguments 

    # Example:
    # {"score_a": a, "score_b": b}

def register_args(parser):
    ...
    # Optional
    # Add command line arguments which will be available in CLI_args

    # Example:
    # parser.add_argument("--my-arg", type=int, required=True)
