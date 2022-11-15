import datetime
import astral.geocoder
import astral.sun

from ..capture.session import Device


city = astral.geocoder.lookup('Bern', astral.geocoder.database())


def get_session_date(session_id):
    device = Device.from_id(session_id)
    if device == Device.HOLOLENS:
        date = session_id.split('_', 1)[1].rsplit('.', 1)[0]
        date = datetime.datetime.strptime(date, '%Y-%m-%d-%H-%M-%S-%f')
        date = date.replace(tzinfo=datetime.timezone.utc)  # is in UTC
        date = date.astimezone(city.tzinfo)  # convert to local time
    elif device == Device.PHONE:
        date = session_id.split('_', 1)[1].rsplit('_', 1)[0]
        date = datetime.datetime.strptime(date, '%Y-%m-%d_%H.%M.%S')
        date = city.tzinfo.localize(date)
    elif device == Device.NAVVIS:
        date = datetime.datetime.strptime(session_id, '%Y-%m-%d_%H.%M.%S')
        date = city.tzinfo.localize(date)
    else:
        raise ValueError(device, session_id)
    return date


def is_session_night(session_id, slack_minutes=15):
    date = get_session_date(session_id)
    if Device.from_id(session_id) == Device.HOLOLENS and date.year in (2020, 2021):
        return False  # the time is incorrect for older sessions
    sun = astral.sun.sun(city.observer, date=date.date(), tzinfo=city.timezone)
    slack = datetime.timedelta(minutes=slack_minutes)
    sunrise = sun['sunrise'] - slack
    sunset = sun['sunset'] + slack
    is_day = sunrise < date < sunset
    return not is_day
