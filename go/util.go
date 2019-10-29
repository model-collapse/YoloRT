package main

//#include "api.h"
import "C"

type LabeledPerson struct {
	Loc        []int32  `json:"loc"`
	Activities []string `json:"activities"`
}

func parseDetectionResultFromC(ctx *C.context_t) (ret []LabeledPerson) {
	for i := 0; i < int(ctx.num_boxes); i++ {
		p := LabeledPerson{Loc: make([]int32, 4)}
		p.Loc[0] = int32(ctx.boxes[i].left)
		p.Loc[1] = int32(ctx.boxes[i].top)
		p.Loc[2] = int32(ctx.boxes[i].width)
		p.Loc[3] = int32(ctx.boxes[i].height)

		for j := 0; j < int(C.NUM_ACTIVITIES) && ctx.activities[i][j] != (*C.char)(nil); j++ {
			actName := C.GoString(ctx.activities[i][j])
			p.Activities = append(p.Activities, actName)
		}

		ret = append(ret, p)
	}

	return
}
