
scorefxn_name = "len_af3"
description = "Returns the number of atoms in the aa_af3 AtomArray (total rows)."

def score(aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args):
    """
    Simple scorer: returns the length (number of rows) of aa_af3.
    Signature: (aa_design, aa_af3, meta, confidences) -> dict[str, scalar|array]
    """
    
    n = int(aa_af3.shape[0])

    return {"len_af3": n}
