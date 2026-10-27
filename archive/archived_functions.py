def find_default_piezo_position(api, mqtt, default_headweight, default_gw):
    """ Discovering default piezo position from lowering the head onto the loadcell """
    #put cutter in maintenance mode
    mqtt.command("mode_maintenance")
    #TODO is there a way to check in maintenance mode
    time.sleep(2)

    #drop head
    api.set("head_ctrl", 1)
    #TODO is there a way to monitor when the head is down?
    time.sleep(5)

    #move piezo until default headweight is achieved
    headweight_vs_piezo = []
    for pos in range(0.6, 3.1, 0.2):
        move_piezo(api, mqtt, pos)
        time.sleep(20)
        headweight = float(mqtt.monitor("head_weight"))
        headweight_vs_piezo.append([headweight, pos])

    h, p = np.array(headweight_vs_piezo).T
    fit = stats.linregress(h, p)

    a_known = default_gw
    default_piezo_pos = fit.slope * a_known + fit.intercept
    return default_piezo_pos