import original from '../../../../vite.config.ts';
import { fileURLToPath } from 'node:url';
export default env=>{
 const config=typeof original==='function'?original({...env,mode:'independent-review'}):original;
 return{...config,resolve:{...config.resolve,alias:{...config.resolve?.alias,'@mediapipe/tasks-vision':fileURLToPath(new URL('./independentSdk.mjs',import.meta.url))}}};
};
