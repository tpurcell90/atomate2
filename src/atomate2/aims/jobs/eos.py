"""Defines the jobs for EOS needed by the QHA workflows."""

from dataclasses import dataclass, field

from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.io.aims.sets.core import RelaxSetGenerator

from atomate2.aims.jobs.core import RelaxMaker


# No prefix, base atomate 2 parameters
@dataclass
class EosRelaxMaker(RelaxMaker):
    """
    Maker to create FHI-aims relaxation job using EOS parameters.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .AimsInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_aims_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.copy_aims_outputs`.
    run_aims_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.run_aims`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    store_output_data: bool
        Whether the job output (TaskDoc) should be stored in the JobStore through
        the response.
    """

    name: str = "EOS GGA relax"
    input_set_generator: AimsInputGenerator = field(default_factory=RelaxSetGenerator)
