function pezdispenser_switch_to_txt2img(text) {
    switch_to_txt2img()
    return text
}

function pezdispenser_switch_to_img2img(text) {
    switch_to_img2img()
    return text
}

function pezdispenser_show_progress_buttons(show) {
    gradioApp().getElementById("pezdispenser_interrupt_button").style.display = show ? "block" : "none"
}

function pezdispenser_submit() {
    pezdispenser_show_progress_buttons(true)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById("pezdispenser_results_column"), null, function() {
        pezdispenser_show_progress_buttons(false)
    })

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

onUiLoaded(function() {
    e = gradioApp().querySelector('#pezdispenser_unload_model_button')
    if (e !== null)
        e.title = "Unload model"

    e = gradioApp().querySelector('#pezdispenser_script_unload_model_button')
    if (e !== null)
        e.title = "Unload model"
})
