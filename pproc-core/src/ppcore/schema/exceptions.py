class PProcSchemaError(Exception):
    pass


class PProcConfigSchemaError(PProcSchemaError):
    pass


class PProcInputSchemaError(PProcSchemaError):
    pass


class PProcStepSchemaError(PProcSchemaError):
    pass


class PProcDatasetError(Exception):
    pass
