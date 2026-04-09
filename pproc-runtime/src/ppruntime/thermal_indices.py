from earthkit.data import FieldList
import earthkit.meteo.solar
import thermofeel
from meters import metered

from earthkit.workflows.backends.earthkit import Metadata, resolve_metadata
from ppruntime.thermo.helpers import (
    compute_ehPa,
    field_values,
    find_utci_missing_values,
    get_datetime,
    latlon,
    step_interval,
    units,
    validate_utci,
    create_output, 
    create_surface_output,
)

logger = logging.getLogger(__name__)


def _check_shape(fields: earthkit.data.FieldList, params: list[str]):
    shape = None
    for param in params:
        selected = fields.sel(param=param)
        if len(selected) == 0:
            raise ValueError(
                f"Field {param} not found in fields: \n {fields.ls(namespace='mars')}"
            )
        if shape is None:
            shape = selected.values.shape
        assert (
            shape == selected.values.shape
        ), f"Shape mismatch for {param} {shape} != {selected.values.shape}"


@metered("cossza", out=logger.debug)
def calc_cossza(*fields: FieldList, metadata: Metadata = None) -> FieldList:
    fields = sum(fields[1:], fields[0])
    lats, lons = latlon(fields)

    basetime, validtime = get_datetime(fields)

    delta = step_interval(fields)

    dtbegin = validtime - timedelta(hours=delta)
    dtend = validtime

    cossza = earthkit.meteo.solar.cos_solar_zenith_angle_integrated(
        latitudes=lats,
        longitudes=lons,
        begin_date=dtbegin,
        end_date=dtend,
        integration_order=2,
    )

    return create_output(
        cossza, metadata_intensity(fields)[:1], {**resolve_metadata(metadata), "paramId": "214001"}
    )


@metered("hmdx", out=logger.debug)
def calc_hmdx(*fields: FieldList, metadata: Metadata = None) -> FieldList:
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2d"])
    t2m = field_values(fields, "2t")  # Kelvin
    td = field_values(fields, "2d")  # Kelvin

    hmdx = thermofeel.calculate_humidex(t2_k=t2m, td_k=td)  # Kelvin

    return create_surface_output(
        hmdx, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261016"}
    )


@metered("rhp", out=logger.debug)
def calc_rhp(*fields: FieldList, metadata: Metadata = None) -> FieldList:
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2d"])
    t2m = field_values(fields, "2t")  # Kelvin
    td = field_values(fields, "2d")  # Kelvin

    rhp = thermofeel.calculate_relative_humidity_percent(t2_k=t2m, td_k=td)  # %

    return create_surface_output(rhp, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "260242"})


@metered("heatx", out=logger.debug)
def calc_heatx(*fields: FieldList, metadata: Metadata = None) -> FieldList:
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2d"])

    t2m = field_values(fields, "2t")  # Kelvin
    td = field_values(fields, "2d")  # Kelvin

    heatx = thermofeel.calculate_heat_index_adjusted(t2_k=t2m, td_k=td)  # Kelvin

    return create_surface_output(
        heatx, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "260004"}
    )


@metered("dsrp", out=logger.debug)
def calc_dsrp(*fields: FieldList, metadata: Metadata = None):
    """
    In the absence of dsrp, approximate it with fdir and cossza.
    Note this introduces some amount of error as cossza approaches zero
    """
    fields = sum(fields[1:], fields[0])
    
    # Will use dsrp if available, otherwise approximate it
    if "dsrp" in fields.indices()["param"]:
        return fields.sel(param="dsrp")

    fdir = field_values(fields, "fdir")  # W/m2
    cossza = field_values(fields, "cossza")

    dsrp = thermofeel.approximate_dsrp(fdir, cossza)

    return create_output(dsrp, metadata_accumulation(fields), {**resolve_metadata(metadata), "paramId": "47"})


@metered("utci", out=logger.debug)
def calc_utci(*fields: FieldList, metadata: Metadata = None, validate=True):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2d", "10si", "mrt"])
    lats, lons = latlon(fields)

    t2m = field_values(fields, "2t")  # Kelvin
    t2d = field_values(fields, "2d")  # Kelvin
    ws = field_values(fields, "10si")  # m/s
    mrt = field_values(fields, "mrt")  # Kelvin

    ehPa = compute_ehPa(t2m, t2d)

    utci = thermofeel.calculate_utci(t2_k=t2m, va=ws, mrt=mrt, ehPa=ehPa)  # Kelvin

    for index in range(len(utci)):
        missing = find_utci_missing_values(
            t2m[index],
            ws[index],
            mrt[index],
            ehPa[index],
            utci[index],
            print_misses,
        )

        if validate:
            validate_utci(utci[index], missing, lats, lons)

        utci[index][missing] = np.nan

    return create_surface_output(
        utci, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261001"}
    )


@metered("wbgt", out=logger.debug)
def calc_wbgt(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2d", "10si", "mrt"])
    t2m = field_values(fields, "2t")  # Kelvin
    t2d = field_values(fields, "2d")  # Kelvin
    ws = field_values(fields, "10si")  # m/s
    mrt = field_values(fields, "mrt")  # Kelvin

    wbgt = thermofeel.calculate_wbgt(t2m, mrt, ws, t2d)  # Kelvin

    return create_surface_output(
        wbgt, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261014"}
    )


@metered("gt", out=logger.debug)
def calc_gt(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "10si", "mrt"])
    t2m = field_values(fields, "2t")  # Kelvin
    ws = field_values(fields, "10si")  # m/s
    mrt = field_values(fields, "mrt")  # Kelvin

    gt = thermofeel.calculate_bgt(t2m, mrt, ws)  # Kelvin

    return create_surface_output(
        gt, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261015"}
    )

@metered("wbt", out=logger.debug)
def calc_wbt(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "2r"])
    t2m = field_values(fields, "2t")  # Kelvin
    rhp = field_values(fields, "2r")  # %

    wbt = thermofeel.calculate_wbt(t2_k=t2m, rh=rhp)  # Kelvin

    return create_surface_output(
        wbt, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261023"}
    )

@metered("nefft", out=logger.debug)
def calc_nefft(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "10si", "2r"])
    t2m = field_values(fields, "2t")  # Kelvin
    ws = field_values(fields, "10si")  # m/s
    rhp = field_values(fields, "2r")  # %

    nefft = thermofeel.calculate_normal_effective_temperature(
        t2m, ws, rhp
    )  # Kelvin

    return create_surface_output(
        nefft, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261018"}
    )


@metered("wcf", out=logger.debug)
def calc_wcf(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "10si"])
    t2m = field_values(fields, "2t")  # Kelvin
    ws = field_values(fields, "10si")  # m/s

    wcf = thermofeel.calculate_wind_chill(t2m, ws)  # Kelvin

    return create_surface_output(
        wcf, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "260005"}
    )


@metered("aptmp", out=logger.debug)
def calc_aptmp(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["2t", "10si", "2r"])
    t2m = field_values(fields, "2t")  # Kelvin
    ws = field_values(fields, "10si")  # m/s
    rhp = field_values(fields, "2r")  # %

    aptmp = thermofeel.calculate_apparent_temperature(
        t2_k=t2m, va=ws, rh=rhp
    )  # Kelvin

    return create_surface_output(
        aptmp, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "260255"}
    )


@metered("mrt", out=logger.debug)
def calc_mrt(*fields: FieldList, metadata: Metadata = None):
    fields = sum(fields[1:], fields[0])
    _check_shape(fields, ["ssrd", "fdir", "strd", "ssr", "dsrp", "str", "cossza"])
    cossza = field_values(fields, "cossza")  
    dsrp = field_values(fields, "dsrp") 

    delta = step_interval(fields)
    seconds_in_time_step = delta * 3600  # steps are in hours

    f = 1.0 / float(seconds_in_time_step)

    ssrd = field_values(fields, "ssrd")  # W/m2
    fdir = field_values(fields, "fdir")  # W/m2
    strd = field_values(fields, "strd")  # W/m2
    strr = field_values(fields, "str")  # W/m2
    ssr = field_values(fields, "ssr")  # W/m2

    # remove negative values from deaccumulated solar fields
    for v in ssrd, fdir, strd, ssr:
        v[v < 0] = 0

    mrt = thermofeel.calculate_mean_radiant_temperature(
        ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
    )  # Kelvin

    return create_surface_output(
        mrt, metadata_intensity(fields), {**resolve_metadata(metadata), "paramId": "261002"}
    )