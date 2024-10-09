import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
  name: "PointAgiClub.imgTiler",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PC TilerImage") {
      console.log("beforeRegisterNodeDef", nodeData);
      var input_name = "tile";

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (
        type,
        index,
        connected,
        link_info
      ) {
        console.log(arguments);
        window.aaa = this;

        if (!link_info || link_info?.type !== "SelectTile") return;

        let slot_i = 1;
        for (let i = 0; i < this.inputs.length; i++) {
          let input_i = this.inputs[i];
          if (input_i.name.match("tile")) {
            input_i.name = `${input_name}${slot_i}`;
            slot_i++;
          }
        }

        const converted_count = 2;

        if (!connected && this.inputs.length > converted_count) {
          const stackTrace = new Error().stack;
          if (
            !stackTrace.includes("LGraphNode.prototype.connect") &&
            !stackTrace.includes("LGraphNode.connect") &&
            !stackTrace.includes("loadGraphData") &&
            this.inputs[index].name.match("tile")
          ) {
            this.removeInput(index);
          }
        }

        let last_slot = this.inputs[this.inputs.length - 1];
        console.log(JSON.parse(JSON.stringify(last_slot)));
        if (last_slot.link != undefined) {
          this.addInput(
            `${input_name}${slot_i}`,
            this.inputs.find((item) => item.name.match("tile")).type
          );
        }
      };
    }
  },
});
