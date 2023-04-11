from datetime import timedelta


def climatology_date(fc_date):

    weekday = fc_date.weekday()

    # tuesday to thursday -> take previous monday clim, else previous thursday clim
    if weekday == 0 or weekday > 3:
        clim_date = fc_date - timedelta(days=(weekday + 4) % 7)
    else:
        clim_date = fc_date - timedelta(days=weekday)

    return clim_date
