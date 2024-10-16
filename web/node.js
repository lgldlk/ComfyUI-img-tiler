import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
  name: "PointAgiClub.imgTiler",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PC TilerImage") {
      var input_name = "tile";

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (
        type,
        index,
        connected,
        link_info
      ) {
        if (!link_info || link_info?.type !== "Pc_Tiles") return;
        const tileInput = this.inputs.filter((item) => item.name.match("tile"));
        if (!connected && tileInput.length > 1) {
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
        let slot_i = 1;

        tileInput.map((item) => {
          item.name = `${input_name}${slot_i++}`;
        });

        let last_slot = this.inputs[this.inputs.length - 1];
        console.log(JSON.parse(JSON.stringify(last_slot)));
        if (last_slot.link != undefined) {
          this.addInput(`${input_name}${slot_i}`, tileInput[0].type);
        }
      };
    }
  },
});
