// Exercise model switching and rendering without a browser dependency.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor() { this.children=[]; this.events={}; this.value=''; this.classList={toggle(){},add(){},remove(){}}; }
  addEventListener(name, handler) { this.events[name]=handler; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  setAttribute() {}
  querySelector() { return new Element(); }
  getContext() { return {drawImage(){}}; }
}
const ids=[...fs.readFileSync('web/index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
elements.model.value='clip'; elements.confidence.value='40';
const base={original_image:'data:image/jpeg;base64,',width:100,height:100,original_width:100,original_height:100,elapsed_ms:100};
const clip={...base,model:'CLIP ViT-B/32',analysis_type:'classification',predictions:Array.from({length:42},(_,i)=>({tile:'1s',name:'1 bamboo',score:.02}))};
const yolo={...base,model:'AR-Mahjong YOLOv8n',analysis_type:'detection',annotated_image:'data:image/jpeg;base64,',detections:[{tile:'4S',name:'Winter',confidence:.8606,xyxy:[0,0,80,80]}]};
const context=vm.createContext({document:{getElementById:id=>{assert(elements[id],id);return elements[id];},createElement:()=>new Element()},window:{addEventListener(){}},Image:class{async decode(){}},Blob,URL,
 fetch:async url=>({ok:true,json:async()=>url==='/api/prepare'?base:url.includes('model=clip')?clip:yolo})});
vm.runInContext(fs.readFileSync('web/app.js','utf8'),context);
(async()=>{
 await vm.runInContext("selectFile({name:'tile.heic',type:'image/heic',size:100})",context);
 elements.model.events.change();
 await elements.analyze.onclick();
 assert.equal(elements.tiles.children.length,5);
 assert.equal(elements['result-title'].textContent,'1 bamboo');
 assert(elements['download-image'].hidden);
 assert(elements['annotated-tab'].disabled);
 assert(elements['score-note'].textContent.includes('relative softmax'));
 elements.model.value='yolo'; elements.model.events.change();
 assert(elements.results.hidden);
 await elements.analyze.onclick();
 assert.equal(elements['result-title'].textContent,'Winter');
 assert.equal(elements.tiles.children.length,1);
 assert(!elements['download-image'].hidden);
 assert(!elements['annotated-tab'].disabled);
 console.log('Model switching, CLIP ranking, and YOLO rendering passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
