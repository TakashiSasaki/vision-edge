import path from 'node:path';
export function requestedDelegate(value=process.env.REAL_SDK_DELEGATE) {
  if(value===undefined)return 'GPU';
  if(value!=='CPU'&&value!=='GPU')throw new Error('INVALID_REAL_SDK_DELEGATE: expected explicit CPU or GPU');
  return value;
}
export function commandDelegate(args) {
  if(!args.length)return 'GPU'; // Ambient environment cannot change the default command.
  if(args.length!==1||!['--delegate=CPU','--delegate=GPU'].includes(args[0]))throw new Error('INVALID_REAL_SDK_DELEGATE: use --delegate=CPU or --delegate=GPU');
  return requestedDelegate(args[0].slice('--delegate='.length));
}
export function delegatePaths(delegate) {
  requestedDelegate(delegate);const cpu=delegate==='CPU';
  return {port:cpu?4176:4175,origin:`http://127.0.0.1:${cpu?4176:4175}`,build:cpu?'.cache/vision-edge-real-sdk-cpu':'.cache/vision-edge-real-sdk',runs:cpu?'.cache/real-sdk-cpu-runs':'.cache/real-sdk-runs',last:cpu?'.cache/real-sdk-cpu-last-run.txt':'.cache/real-sdk-last-run.txt',probe:cpu?'.cache/pr07-native-probe-cpu':'.cache/pr07-native-probe'};
}
// Test-build-only variant of the same engine. Observers never rewrite SDK args.
export function cpuEngineVariant(code,id) {
  if(path.basename(id)!=='visionEngine.ts')return undefined;
  const anchor="const DELEGATE = 'GPU' as const;";
  if(code.split(anchor).length!==2)throw new Error('CPU verification engine anchor changed; inspect before running');
  return code.replace(anchor,"const DELEGATE = 'CPU' as const;");
}
