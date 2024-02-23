import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: root_window

    signal aiModelToggled(strmodelName:string)
    property var processedFeeds: []
    property var imageFeeds: []

    function close_system()
    {
        root_window.close()
        Qt.quit()
    }

    visible: true
    // flags: App.fullscreen ? Qt.Window | Qt.CustomizeWindowHint : Qt.Window
    // visibility: App.fullscreen ? "FullScreen" : "Maximized"

    width: 1280
    height: 900

    Label {
        id: fpsLabel
        anchors {
            bottom: parent.bottom
            horizontalCenter: parent.horizontalCenter
            margins: 10
        }

        font {
            family: "Arial"
            pixelSize: 18
        }

        Connections {
            target: main_controller

            function onFpsCounterUpdated() {
                fpsLabel.text = `${main_controller.get_fps_count()}`                
            }
        }
    }


    Column {
        Row {
            width: root_window.width
            height: root_window.height * 0.45 - 30
            Item {
                id: camerasItem
                width: root_window.width * 1
                height: root_window.height * 0.45 - 30

                Row {
                    anchors {
                        fill: parent
                        margins: 10
                    }
                    spacing: 10
                    Repeater {
                        id: feedRepeater
                        model: ["image", "depth"]
                        delegate: Rectangle {
                            width: camerasItem.width / 2 - 15
                            height: camerasItem.height
                            color: "black"

                            Image {
                                id: feed
                                anchors {
                                    fill: parent
                                    margins: 2
                                }
                                source: `image://provider/none/`
                            }

                            Component.onCompleted: {
                                root_window.imageFeeds.push(feed)
                            }
                        }
                    }
                }
            }        

            Item {
                width: root_window.width * 0.2
                height: root_window.height * 0.45 - 30
                visible: false

                Label {
                    id: statesTitleLabel
                    anchors {
                        top: parent.top
                        topMargin: 10
                        horizontalCenter: parent.horizontalCenter
                    }

                    text: "States"
                    font {
                        bold: true
                        pixelSize: 18
                    }
                }

                Label {
                    id: statesCurrentStateLabel
                    anchors {
                        top: statesTitleLabel.bottom
                        topMargin: 4
                        horizontalCenter: parent.horizontalCenter
                    }

                    text: main_controller.get_current_state()
                    font {
                        pixelSize: 12
                    }

                    Connections {
                        target: main_controller

                        function onStateSwitched() {
                            statesCurrentStateLabel.text = main_controller.get_current_state()

                        }
                    }
                }

                Column {
                    spacing: 10
                    anchors.centerIn: parent

                    Repeater {
                        model: main_controller.get_state_names()
                        delegate: Button {
                            text: modelData
                            width: 180
                            anchors.horizontalCenter: parent.horizontalCenter
                            onClicked: {
                                main_controller.switch_state(modelData)
                            }
                        }
                    }
                }
            }
        }

        Item {
            height: 120
            width: root_window.width

            Row {
                spacing: 10
                anchors.centerIn: parent

                Repeater {
                    model: main_controller.get_model_names()
                    delegate: Item {
                        id: aiModelStatusItem
                        width: (processedFeedsItem.width - (10 * (processedFeedRepeater.model.length - 1)) - 20) * (1 / processedFeedRepeater.model.length)
                        height: processedFeedsItem.height - 20
                        Column {
                            anchors.centerIn: parent
                            spacing: 15
                            Label {
                                id: statusLabel
                                anchors.horizontalCenter: parent.horizontalCenter
                                font {
                                    family: "Arial"
                                    pixelSize: 14
                                }
                                text: `Status: ${main_controller.is_ai_model_running(modelData) ? "RUNNING" : "STOPPED"}`
                                color: "red"

                                Connections {
                                    target: root_window

                                    function onAiModelToggled(modelName) {
                                        if (modelName === modelData) {
                                            statusLabel.text = `Status: ${main_controller.is_ai_model_running(modelData) ? "RUNNING" : "STOPPED"}`
                                            statusLabel.color = main_controller.is_ai_model_running(modelData) ? "green" : "red"
                                        }
                                    }
                                }
                            }
                            Button {
                                text: modelData
                                width: aiModelStatusItem.width * 0.75
                                anchors.horizontalCenter: parent.horizontalCenter
                                onClicked: {
                                    main_controller.toggle_ai_model(modelData)
                                    root_window.aiModelToggled(modelData)
                                }
                            }
                        }
                    }
                }
            }
        }

        Item {
            id: processedFeedsItem
            height: root_window.height  - (root_window.height * 0.45 + 120)
            width: root_window.width

            Row {
                anchors {
                    fill: parent
                    margins: 10
                }
                spacing: 10
                
                Repeater {
                    id: processedFeedRepeater
                    model: main_controller.get_model_names()
                    delegate: Item {
                        width: (processedFeedsItem.width - (10 * (processedFeedRepeater.model.length - 1)) - 20) * (1 / processedFeedRepeater.model.length)
                        height: processedFeedsItem.height - 20

                        Rectangle {
                            anchors {
                                fill: parent
                            }
                            color: "black"

                            Label {
                                id: disabledLabel
                                anchors.centerIn: parent
                                font {
                                    family: "Arial"
                                    pixelSize: 16
                                }
                                color: "white"
                                text: `${modelData} is disabled.`
                            }

                            Image {
                                id: processedFeed
                                property string text: "NO TEXT"

                                anchors {
                                    fill: parent
                                    margins: 2
                                }
                                // source: `image://provider/none/`
                                visible: false

                                Connections {
                                    target: root_window

                                    function onAiModelToggled(modelName) {
                                        if (modelName === modelData) {
                                            let isModelRunning = main_controller.is_ai_model_running(modelData) 
                                            processedFeed.visible = isModelRunning
                                            disabledLabel.visible = !isModelRunning                                       
                                        }
                                    }
                                }

                                Label {
                                    visible: main_controller.get_model_output_type(modelData) === "TEXT"
                                    anchors {
                                        fill: parent
                                        margins: 10
                                    }

                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.WordWrap

                                    font {
                                        family: "Arial"
                                        pixelSize: 28
                                    }

                                    text: processedFeed.text
                                    color: "white"
                                }
                            }  

                            Component.onCompleted: {
                                root_window.processedFeeds.push(processedFeed)
                            } 
                        }
                    }
                }
            }
        }
    }
    
    Timer {
        interval: 100
        repeat: true
        running: true
        onTriggered: {
            main_controller.process_models()
            for (var i = 0; i < root_window.imageFeeds.length; i++) {
                root_window.imageFeeds[i].source = `image://provider/${feedRepeater.model[i]}/${Math.floor(Math.random() * 100)}`
            }
            for (var i = 0; i < root_window.processedFeeds.length; i++) {
                if (root_window.processedFeeds[i].visible) {
                    if (main_controller.get_model_output_type(processedFeedRepeater.model[i]) === "TEXT") {
                        root_window.processedFeeds[i].text = main_controller.get_model_output(processedFeedRepeater.model[i])
                    } else if (main_controller.get_model_output_type(processedFeedRepeater.model[i]) === "IMAGE") {
                        root_window.processedFeeds[i].source = `image://provider/${processedFeedRepeater.model[i]}/${Math.floor(Math.random() * 100)}`
                    }
                }
            }
        }
    }
    
    Timer {
        interval: 50
        repeat: true
        running: true
        onTriggered: {
            main_controller.log_data()
            let models = main_controller.get_model_names()

            for (let i = 0; i < models.length; i++) {
                let model = models[i]
                aiModelToggled(model)
            }
        }
    }
}
