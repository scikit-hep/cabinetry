import logging

import cabinetry


def test_set_logging(caplog):
    log = logging.getLogger("cabinetry")

    # message not recorded by default
    log.debug("log message")
    assert len(caplog.records) == 0
    caplog.clear()

    # set custom logging, now message is recorded
    cabinetry.set_logging()
    log.debug("log message")
    assert "log message" in [rec.message for rec in caplog.records]
