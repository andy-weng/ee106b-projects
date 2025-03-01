
"use strict";

let AddCO2Source = require('./AddCO2Source.js')
let AddRfidTag = require('./AddRfidTag.js')
let AddSoundSource = require('./AddSoundSource.js')
let AddThermalSource = require('./AddThermalSource.js')
let DeleteCO2Source = require('./DeleteCO2Source.js')
let DeleteRfidTag = require('./DeleteRfidTag.js')
let DeleteSoundSource = require('./DeleteSoundSource.js')
let DeleteThermalSource = require('./DeleteThermalSource.js')
let LoadExternalMap = require('./LoadExternalMap.js')
let LoadMap = require('./LoadMap.js')
let MoveRobot = require('./MoveRobot.js')
let RegisterGui = require('./RegisterGui.js')

module.exports = {
  AddCO2Source: AddCO2Source,
  AddRfidTag: AddRfidTag,
  AddSoundSource: AddSoundSource,
  AddThermalSource: AddThermalSource,
  DeleteCO2Source: DeleteCO2Source,
  DeleteRfidTag: DeleteRfidTag,
  DeleteSoundSource: DeleteSoundSource,
  DeleteThermalSource: DeleteThermalSource,
  LoadExternalMap: LoadExternalMap,
  LoadMap: LoadMap,
  MoveRobot: MoveRobot,
  RegisterGui: RegisterGui,
};
